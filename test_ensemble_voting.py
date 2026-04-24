"""
Test script to evaluate ensemble voting vs single model performance.
This script loads multiple GNN models and compares:
1. Single GINE model accuracy
2. Ensemble voting (majority + weighted) accuracy
3. Inference time comparison
"""

import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import json
import time
from collections import Counter

# Import model classes
from train_multi_gnn_realistic import GINEModel, GraphSAGEModel, TransformerConvModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Load hyperparameters from JSON files
print("📋 Loading model hyperparameters...")
with open('gine_best_hyperparameters.json', 'r') as f:
    gine_params = json.load(f)
with open('graphsage_best_hyperparameters.json', 'r') as f:
    graphsage_params = json.load(f)
with open('transformerconv_best_hyperparameters.json', 'r') as f:
    transformerconv_params = json.load(f)

print(f"GINE params: hidden={gine_params['hidden_channels']}, dropout={gine_params['dropout']}")
print(f"GraphSAGE params: hidden={graphsage_params['hidden_channels']}, dropout={graphsage_params['dropout']}")
print(f"TransformerConv params: hidden={transformerconv_params['hidden_channels']}, dropout={transformerconv_params['dropout']}, heads={transformerconv_params.get('heads', 8)}\n")

# Load models with correct hyperparameters
print("📦 Loading trained models...")

# GINE Model
print("  Loading GINE model...")
gine_model = GINEModel(
    in_channels=11, 
    edge_dim=4, 
    hidden_channels=gine_params['hidden_channels'], 
    dropout=gine_params['dropout'], 
    num_classes=3
)
gine_model.load_state_dict(torch.load('best_gine_model.pt', map_location=device))
gine_model.to(device)
gine_model.eval()

# GraphSAGE Model
print("  Loading GraphSAGE model...")
graphsage_model = GraphSAGEModel(
    in_channels=11, 
    hidden_channels=graphsage_params['hidden_channels'], 
    dropout=graphsage_params['dropout'], 
    num_classes=3
)
graphsage_model.load_state_dict(torch.load('best_graphsage_model.pt', map_location=device))
graphsage_model.to(device)
graphsage_model.eval()

# TransformerConv Model
print("  Loading TransformerConv model...")
transformerconv_model = TransformerConvModel(
    in_channels=11, 
    edge_dim=4, 
    hidden_channels=transformerconv_params['hidden_channels'], 
    num_heads=transformerconv_params.get('heads', 8),  # Use num_heads parameter
    dropout=transformerconv_params['dropout'], 
    num_classes=3
)
transformerconv_model.load_state_dict(torch.load('best_transformerconv_model.pt', map_location=device))
transformerconv_model.to(device)
transformerconv_model.eval()

print("✅ All models loaded successfully!\n")

# Load test graph
print("📊 Loading test graph...")
graph_data = torch.load('supply_chain_graph.pt', weights_only=False)
graph_data = graph_data.to(device)
print(f"Graph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges\n")

# Model accuracies for weighted voting
model_accuracies = {
    'GINE': 0.7779,
    'GraphSAGE': 0.7775,
    'TransformerConv': 0.7765
}

def predict_single_model(model, data, model_name=''):
    """Get predictions from a single model."""
    with torch.no_grad():
        # Check if model uses edge attributes
        if model_name in ['GINE', 'TransformerConv'] and hasattr(data, 'edge_attr') and data.edge_attr is not None:
            out = model(data.x, data.edge_index, data.edge_attr)
        else:
            # GraphSAGE and other models don't use edge attributes
            out = model(data.x, data.edge_index)
        preds = out.argmax(dim=1)
        probs = torch.exp(out)  # Convert log_softmax to probabilities
    return preds, probs

def ensemble_majority_voting(predictions_list):
    """Simple majority voting - each model gets 1 vote."""
    num_nodes = predictions_list[0].shape[0]
    ensemble_preds = torch.zeros(num_nodes, dtype=torch.long, device=device)
    
    for i in range(num_nodes):
        votes = [preds[i].item() for preds in predictions_list]
        ensemble_preds[i] = Counter(votes).most_common(1)[0][0]
    
    return ensemble_preds

def ensemble_weighted_voting(predictions_list, probs_list, weights):
    """Weighted voting based on model accuracy."""
    num_nodes = predictions_list[0].shape[0]
    num_classes = probs_list[0].shape[1]
    
    # Weighted average of probabilities
    weighted_probs = torch.zeros(num_nodes, num_classes, device=device)
    for probs, weight in zip(probs_list, weights):
        weighted_probs += probs * weight
    
    ensemble_preds = weighted_probs.argmax(dim=1)
    return ensemble_preds

def calculate_accuracy(predictions, labels):
    """Calculate accuracy."""
    correct = (predictions == labels).sum().item()
    total = labels.shape[0]
    return correct / total

# Run predictions
print("🔮 Running predictions...\n")

# Single model predictions
print("1️⃣ Single Model Predictions:")
start_time = time.time()
gine_preds, gine_probs = predict_single_model(gine_model, graph_data, 'GINE')
gine_time = time.time() - start_time
print(f"   GINE: {gine_time*1000:.2f}ms")

start_time = time.time()
graphsage_preds, graphsage_probs = predict_single_model(graphsage_model, graph_data, 'GraphSAGE')
graphsage_time = time.time() - start_time
print(f"   GraphSAGE: {graphsage_time*1000:.2f}ms")

start_time = time.time()
transformerconv_preds, transformerconv_probs = predict_single_model(transformerconv_model, graph_data, 'TransformerConv')
transformerconv_time = time.time() - start_time
print(f"   TransformerConv: {transformerconv_time*1000:.2f}ms\n")

# Ensemble predictions
print("2️⃣ Ensemble Predictions:")
predictions_list = [gine_preds, graphsage_preds, transformerconv_preds]
probs_list = [gine_probs, graphsage_probs, transformerconv_probs]
weights = [model_accuracies['GINE'], model_accuracies['GraphSAGE'], model_accuracies['TransformerConv']]
weights = [w / sum(weights) for w in weights]  # Normalize weights

start_time = time.time()
majority_preds = ensemble_majority_voting(predictions_list)
majority_time = time.time() - start_time
print(f"   Majority Voting: {majority_time*1000:.2f}ms")

start_time = time.time()
weighted_preds = ensemble_weighted_voting(predictions_list, probs_list, weights)
weighted_time = time.time() - start_time
print(f"   Weighted Voting: {weighted_time*1000:.2f}ms\n")

# Calculate accuracies if labels are available
if hasattr(graph_data, 'y') and graph_data.y is not None:
    print("📈 Accuracy Comparison:")
    print(f"   GINE (single):           {calculate_accuracy(gine_preds, graph_data.y)*100:.2f}%")
    print(f"   GraphSAGE (single):      {calculate_accuracy(graphsage_preds, graph_data.y)*100:.2f}%")
    print(f"   TransformerConv (single): {calculate_accuracy(transformerconv_preds, graph_data.y)*100:.2f}%")
    print(f"   Majority Voting:         {calculate_accuracy(majority_preds, graph_data.y)*100:.2f}%")
    print(f"   Weighted Voting:         {calculate_accuracy(weighted_preds, graph_data.y)*100:.2f}%\n")
    
    # Calculate improvement
    gine_acc = calculate_accuracy(gine_preds, graph_data.y)
    majority_acc = calculate_accuracy(majority_preds, graph_data.y)
    weighted_acc = calculate_accuracy(weighted_preds, graph_data.y)
    
    print("📊 Improvement Analysis:")
    print(f"   Majority vs GINE: {(majority_acc - gine_acc)*100:+.2f}%")
    print(f"   Weighted vs GINE: {(weighted_acc - gine_acc)*100:+.2f}%\n")
else:
    print("⚠️ No labels available in graph data - cannot calculate accuracy\n")

# Prediction distribution comparison
print("📊 Prediction Distribution:")
print(f"   GINE:           Failed={int((gine_preds==0).sum())}, Degraded={int((gine_preds==1).sum())}, Normal={int((gine_preds==2).sum())}")
print(f"   GraphSAGE:      Failed={int((graphsage_preds==0).sum())}, Degraded={int((graphsage_preds==1).sum())}, Normal={int((graphsage_preds==2).sum())}")
print(f"   TransformerConv: Failed={int((transformerconv_preds==0).sum())}, Degraded={int((transformerconv_preds==1).sum())}, Normal={int((transformerconv_preds==2).sum())}")
print(f"   Majority:       Failed={int((majority_preds==0).sum())}, Degraded={int((majority_preds==1).sum())}, Normal={int((majority_preds==2).sum())}")
print(f"   Weighted:       Failed={int((weighted_preds==0).sum())}, Degraded={int((weighted_preds==1).sum())}, Normal={int((weighted_preds==2).sum())}\n")

# Inference time comparison
total_ensemble_time = gine_time + graphsage_time + transformerconv_time + majority_time
print("⏱️ Inference Time Comparison:")
print(f"   Single GINE:     {gine_time*1000:.2f}ms")
print(f"   Ensemble Total:  {total_ensemble_time*1000:.2f}ms ({total_ensemble_time/gine_time:.1f}x slower)\n")

# Agreement analysis
print("🤝 Model Agreement Analysis:")
gine_vs_graphsage = (gine_preds == graphsage_preds).sum().item() / len(gine_preds) * 100
gine_vs_transformer = (gine_preds == transformerconv_preds).sum().item() / len(gine_preds) * 100
graphsage_vs_transformer = (graphsage_preds == transformerconv_preds).sum().item() / len(graphsage_preds) * 100
all_agree = ((gine_preds == graphsage_preds) & (gine_preds == transformerconv_preds)).sum().item() / len(gine_preds) * 100

print(f"   GINE vs GraphSAGE:      {gine_vs_graphsage:.1f}% agreement")
print(f"   GINE vs TransformerConv: {gine_vs_transformer:.1f}% agreement")
print(f"   GraphSAGE vs TransformerConv: {graphsage_vs_transformer:.1f}% agreement")
print(f"   All 3 models agree:     {all_agree:.1f}% of predictions\n")

# Recommendation
print("=" * 60)
print("💡 RECOMMENDATION:")
print("=" * 60)
if hasattr(graph_data, 'y') and graph_data.y is not None:
    improvement = max(majority_acc - gine_acc, weighted_acc - gine_acc) * 100
    if improvement > 1.0:
        print(f"✅ USE ENSEMBLE: {improvement:.2f}% accuracy improvement justifies the cost")
    elif improvement > 0.3:
        print(f"⚠️ MARGINAL: {improvement:.2f}% improvement - consider if {total_ensemble_time/gine_time:.1f}x slower is acceptable")
    else:
        print(f"❌ STICK WITH GINE: Only {improvement:.2f}% improvement, not worth {total_ensemble_time/gine_time:.1f}x slower inference")
else:
    print("⚠️ Cannot make recommendation without ground truth labels")
    print(f"   However, ensemble is {total_ensemble_time/gine_time:.1f}x slower than single model")
    print(f"   Models agree {all_agree:.1f}% of the time - high agreement suggests minimal ensemble benefit")

print("=" * 60)
