"""
Debug script to check why model predictions are always the same.
Tests model architecture and data preprocessing.
"""

import torch
import pandas as pd
import numpy as np
from train_multi_gnn_realistic import GINEModel
from torch_geometric.data import Data

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# INPUT: Change this to your company_id
company_id = input("Enter company_id (or press Enter for default): ").strip()

if company_id:
    model_path = f'backend/uploads/{company_id}/best_gine_model.pt'
    node_df = pd.read_csv(f'backend/uploads/{company_id}/nodes.csv')
    edge_df = pd.read_csv(f'backend/uploads/{company_id}/edges.csv')
    print(f"✅ Using company data: {company_id}")
else:
    model_path = 'best_gine_model.pt'
    node_df = pd.read_csv('synthetic_nodes.csv')
    edge_df = pd.read_csv('synthetic_edges.csv')
    print(f"✅ Using default data")

print("=" * 80)
print("DEBUGGING MODEL PREDICTIONS")
print("=" * 80)

# ============================================================================
# TEST 1: Check Model Architecture
# ============================================================================
print("\n[TEST 1] Model Architecture Check")
print("-" * 80)

# Load checkpoint to inspect
checkpoint = torch.load(model_path, map_location='cpu')

# Check if model uses is_disrupted feature
print(f"[OK] Model checkpoint keys: {list(checkpoint.keys())[:5]}...")
print(f"[OK] First conv layer shape: {checkpoint['conv1.nn.0.weight'].shape}")

# Auto-detect input channels from checkpoint
if 'conv1.nn.0.weight' in checkpoint:
    in_channels = checkpoint['conv1.nn.0.weight'].shape[1]
elif 'conv1.lin.weight' in checkpoint:
    in_channels = checkpoint['conv1.lin.weight'].shape[1]
else:
    in_channels = 11  # Default fallback

print(f"[OK] Detected input features: {in_channels}")

# Infer hidden_channels
hidden_channels = checkpoint['conv1.nn.0.weight'].shape[0]
print(f"[OK] Hidden channels: {hidden_channels}")

# Initialize model with detected in_channels
model = GINEModel(in_channels=in_channels, edge_dim=4, hidden_channels=hidden_channels, dropout=0.3, num_classes=3)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

print(f"[OK] Model loaded successfully")
print(f"[OK] Model architecture: {model}")

# ============================================================================
# TEST 2: Check Data Preprocessing
# ============================================================================
print("\n[TEST 2] Data Preprocessing Check")
print("-" * 80)

def create_test_data(disrupted_nodes, severity):
    """Create PyG Data object MATCHING TRAINING FORMAT EXACTLY."""
    num_nodes = len(node_df)
    
    # Base features - DO NOT MODIFY (training doesn't modify them!)
    base_features = torch.tensor(
        node_df[['capacity', 'cost_factor', 'risk_level', 'reliability', 'x', 'y']].values,
        dtype=torch.float
    )
    
    # Tier encoding (4 dimensions)
    tier_encoding = torch.zeros((num_nodes, 4), dtype=torch.float)
    for idx, tier in enumerate(node_df['tier'].values):
        tier_encoding[idx, int(tier)] = 1.0
    
    # Concatenate base + tier: 6 + 4 = 10 features
    base_features_with_tier = torch.cat([base_features, tier_encoding], dim=1)
    
    # is_disrupted flag (binary: 0 or 1) - NO SEVERITY!
    # Training uses ONLY binary flag, not severity value
    is_disrupted = torch.zeros((num_nodes, 1), dtype=torch.float)
    for node_id in disrupted_nodes:
        is_disrupted[int(node_id), 0] = 1.0
    
    # Final concatenation: 10 + 1 = 11 features (matches training!)
    x = torch.cat([base_features_with_tier, is_disrupted], dim=1)
    
    print(f"  DEBUG: Created features with shape {x.shape} (should be [num_nodes, 11])")
    print(f"  DEBUG: Disrupted nodes: {disrupted_nodes}, is_disrupted sum: {is_disrupted.sum().item()}")
    
    # Edge index
    edge_index = torch.tensor(
        np.array([edge_df['source'].values, edge_df['target'].values]),
        dtype=torch.long
    )
    
    # Edge features - must match model's edge_dim
    # Detect edge_dim from model checkpoint
    if 'conv1.lin.weight' in checkpoint:
        edge_dim = checkpoint['conv1.lin.weight'].shape[1]
    else:
        edge_dim = 4  # Default
    
    num_edges = len(edge_df)
    edge_attr = torch.zeros((num_edges, edge_dim), dtype=torch.float)
    
    weights = edge_df['capacity_share'].values if 'capacity_share' in edge_df.columns else edge_df['weight'].values
    
    # Fill first 4 dimensions (standard edge features)
    edge_attr[:, 0] = torch.tensor((weights - weights.mean()) / (weights.std() + 1e-8))
    if edge_dim > 2:
        edge_attr[:, 2] = torch.tensor(weights)
    
    for idx, row in edge_df.iterrows():
        source_idx = int(row['source'])
        target_idx = int(row['target'])
        
        source_cost = float(node_df.iloc[source_idx]['cost_factor'])
        target_cost = float(node_df.iloc[target_idx]['cost_factor'])
        if edge_dim > 1:
            edge_attr[idx, 1] = (source_cost + target_cost) / 2.0
        
        source_risk = float(node_df.iloc[source_idx]['risk_level'])
        target_risk = float(node_df.iloc[target_idx]['risk_level'])
        if edge_dim > 3:
            edge_attr[idx, 3] = (source_risk + target_risk) / 2.0
    
    # If edge_dim > 4, pad with zeros (model expects more dimensions)
    # This happens when training accidentally used wrong edge dimensions
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# Test different scenarios
test_scenarios = [
    {"name": "No Disruption", "nodes": [], "severity": 0.0},
    {"name": "Single Node (Low)", "nodes": [0], "severity": 0.3},
    {"name": "Single Node (High)", "nodes": [0], "severity": 0.9},
    {"name": "Multiple Nodes", "nodes": [0, 5, 10], "severity": 0.7},
    {"name": "Many Nodes", "nodes": list(range(20)), "severity": 0.8},
]

print(f"\nTesting {len(test_scenarios)} scenarios...")
print("-" * 80)

results = []
for scenario in test_scenarios:
    data = create_test_data(scenario['nodes'], scenario['severity'])
    data = data.to(device)
    
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr)
        probs = torch.exp(out)
        preds = out.argmax(dim=1)
    
    failed = int((preds == 0).sum())
    degraded = int((preds == 1).sum())
    normal = int((preds == 2).sum())
    
    results.append({
        'scenario': scenario['name'],
        'disrupted_nodes': len(scenario['nodes']),
        'severity': scenario['severity'],
        'failed': failed,
        'degraded': degraded,
        'normal': normal
    })
    
    print(f"{scenario['name']:20s} | Nodes: {len(scenario['nodes']):3d} | Severity: {scenario['severity']:.1f} | "
          f"Failed: {failed:3d} | Degraded: {degraded:3d} | Normal: {normal:3d}")

# ============================================================================
# TEST 3: Check Feature Importance
# ============================================================================
print("\n[TEST 3] Feature Importance Check")
print("-" * 80)

# Create two identical scenarios except for is_disrupted flag
data_without_flag = create_test_data([0], 0.8)
data_without_flag.x[:, -1] = 0.0  # Remove is_disrupted flag

data_with_flag = create_test_data([0], 0.8)

data_without_flag = data_without_flag.to(device)
data_with_flag = data_with_flag.to(device)

with torch.no_grad():
    out_without = model(data_without_flag.x, data_without_flag.edge_index, data_without_flag.edge_attr)
    out_with = model(data_with_flag.x, data_with_flag.edge_index, data_with_flag.edge_attr)
    
    preds_without = out_without.argmax(dim=1)
    preds_with = out_with.argmax(dim=1)

failed_without = int((preds_without == 0).sum())
failed_with = int((preds_with == 0).sum())

print(f"Without is_disrupted flag: Failed={failed_without}, Degraded={int((preds_without==1).sum())}, Normal={int((preds_without==2).sum())}")
print(f"With is_disrupted flag:    Failed={failed_with}, Degraded={int((preds_with==1).sum())}, Normal={int((preds_with==2).sum())}")
print(f"Difference: {abs(failed_with - failed_without)} nodes changed prediction")

if abs(failed_with - failed_without) < 5:
    print("[WARNING] is_disrupted flag has minimal impact! Model may be ignoring it.")
else:
    print("[OK] is_disrupted flag is being used by the model")

# ============================================================================
# TEST 4: Check Feature Statistics
# ============================================================================
print("\n[TEST 4] Feature Statistics")
print("-" * 80)

# Create data with disruption
data_disrupted = create_test_data([0, 5, 10], 0.8)

print(f"Feature tensor shape: {data_disrupted.x.shape}")
print(f"Feature statistics:")
print(f"  Mean: {data_disrupted.x.mean(dim=0)}")
print(f"  Std:  {data_disrupted.x.std(dim=0)}")
print(f"  Min:  {data_disrupted.x.min(dim=0).values}")
print(f"  Max:  {data_disrupted.x.max(dim=0).values}")

print(f"\nis_disrupted feature (last column):")
print(f"  Sum: {data_disrupted.x[:, -1].sum()} (should be 3)")
print(f"  Unique values: {data_disrupted.x[:, -1].unique()}")

# ============================================================================
# TEST 5: Check Model Output Distribution & Oversmoothing
# ============================================================================
print("\n[TEST 5] Model Output Distribution & Oversmoothing Check")
print("-" * 80)

data_test = create_test_data([0], 0.5)
data_test = data_test.to(device)

# Hook to capture embeddings before final layer
embeddings_before_fc = None
def hook_fn(module, input, output):
    global embeddings_before_fc
    embeddings_before_fc = input[0]  # Capture input to final layer

# Register hook on final layer
hook = model.fc.register_forward_hook(hook_fn)

with torch.no_grad():
    out = model(data_test.x, data_test.edge_index, data_test.edge_attr)
    probs = torch.exp(out)

hook.remove()

# Check for oversmoothing
if embeddings_before_fc is not None:
    embedding_variance = embeddings_before_fc.var(dim=0).mean().item()
    print(f"\n🔍 OVERSMOOTHING CHECK:")
    print(f"  Embedding variance before final layer: {embedding_variance:.6f}")
    
    if embedding_variance < 0.01:
        print(f"  ⚠️  CRITICAL: Severe oversmoothing detected!")
        print(f"      All node embeddings are nearly identical.")
        print(f"      Model has {4} message-passing layers - TOO DEEP for 200 nodes!")
        print(f"      Recommendation: Reduce to 1-2 layers for small graphs.")
    elif embedding_variance < 0.1:
        print(f"  ⚠️  WARNING: Moderate oversmoothing detected.")
        print(f"      Node embeddings are becoming too similar.")
        print(f"      Consider reducing number of GNN layers.")
    else:
        print(f"  ✅ OK: Node embeddings maintain diversity.")

print(f"\nRaw output (logits) statistics:")
print(f"  Mean: {out.mean(dim=0)}")
print(f"  Std:  {out.std(dim=0)}")
print(f"  Min:  {out.min(dim=0).values}")
print(f"  Max:  {out.max(dim=0).values}")

print(f"\nProbability distribution:")
print(f"  Class 0 (Failed):   mean={probs[:, 0].mean():.4f}, std={probs[:, 0].std():.4f}")
print(f"  Class 1 (Degraded): mean={probs[:, 1].mean():.4f}, std={probs[:, 1].std():.4f}")
print(f"  Class 2 (Normal):   mean={probs[:, 2].mean():.4f}, std={probs[:, 2].std():.4f}")

# Check if model is stuck predicting one class
class_predictions = out.argmax(dim=1)
unique_preds, counts = torch.unique(class_predictions, return_counts=True)
print(f"\nPrediction distribution:")
for pred, count in zip(unique_preds, counts):
    print(f"  Class {pred}: {count} nodes ({count/len(class_predictions)*100:.1f}%)")

if len(unique_preds) == 1:
    print("[WARNING] Model is predicting only ONE class! Model has collapsed.")
elif len(unique_preds) == 2:
    print("[WARNING] Model is predicting only TWO classes. Limited diversity.")
else:
    print("[OK] Model is predicting all three classes")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

# Check if predictions vary
pred_counts = [r['failed'] for r in results]
if len(set(pred_counts)) == 1:
    print("[PROBLEM] Predictions are IDENTICAL across all scenarios!")
    print("   -> Model is not responding to input changes")
    print("   -> Likely causes:")
    print("     1. Model collapsed during training (overfitting)")
    print("     2. Model ignoring is_disrupted feature")
    print("     3. Features not properly normalized")
else:
    print("[OK] Predictions vary across scenarios")
    print(f"  Failed count range: {min(pred_counts)} - {max(pred_counts)}")

print("\nRecommendations:")
if len(set(pred_counts)) == 1:
    print("  1. Retrain model with more diverse scenarios")
    print("  2. Check training data has balanced classes")
    print("  3. Verify is_disrupted feature is used during training")
    print("  4. Consider using different model architecture")
else:
    print("  Model appears to be working correctly!")
    print("  If predictions still seem wrong, check:")
    print("  1. Training data quality")
    print("  2. Feature scaling during training vs inference")

print("=" * 80)
