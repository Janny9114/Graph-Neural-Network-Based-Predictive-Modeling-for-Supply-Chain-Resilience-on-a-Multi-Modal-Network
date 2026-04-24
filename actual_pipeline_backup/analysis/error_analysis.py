"""
Error Analysis for GNN Models on Supply Chain Disruption Prediction

This script performs comprehensive error analysis to identify:
1. Consistently misclassified scenarios across all models
2. Error patterns by supply chain regions/sectors
3. Model performance on different disruption types
4. Confusion patterns (which classes are confused with each other)
"""

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import model architectures from train_multi_gnn_realistic.py (the actual trained models)
from train_multi_gnn_realistic import (
    GATModel, GCNModel, GraphSAGEModel, GINModel, 
    TransformerConvModel, GINEModel
)


def load_scenarios(scenario_dir='scenario_graphs_edge_disruptions', max_scenarios=2000):
    """Load scenario data."""
    print(f"\n📂 Loading scenarios from {scenario_dir}...")
    metadata_path = os.path.join(scenario_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    num_scenarios = min(metadata['num_scenarios'], max_scenarios)
    
    data_list = []
    for i in tqdm(range(num_scenarios), desc="Loading"):
        scenario_path = os.path.join(scenario_dir, f'scenario_{i:05d}.pt')
        data = torch.load(scenario_path, weights_only=False)
        data.scenario_id = i  # Add scenario ID for tracking
        data_list.append(data)
    
    print(f"  ✓ Loaded {len(data_list)} scenarios")
    return data_list


def load_trained_model(model_class, model_name, model_path, in_channels, num_classes, device):
    """Load a trained model from checkpoint."""
    # Create model with EXACT hyperparameters from tuning results
    # Updated with best hyperparameters from tune_all_gnn_hyperparameters.py
    if model_name == 'GAT':
        # From gat_best_hyperparameters.json
        model = model_class(in_channels, hidden_channels=256, num_heads=4, dropout=0.1, num_classes=num_classes)
    elif model_name == 'GCN':
        # From gcn_best_hyperparameters.json
        model = model_class(in_channels, hidden_channels=256, dropout=0.1, num_classes=num_classes)
    elif model_name == 'GraphSAGE':
        # From graphsage_best_hyperparameters.json
        model = model_class(in_channels, hidden_channels=256, dropout=0.1, num_classes=num_classes)
    elif model_name == 'GIN':
        # From gin_best_hyperparameters.json
        model = model_class(in_channels, hidden_channels=256, dropout=0.2, num_classes=num_classes)
    elif model_name == 'TransformerConv':
        # From transformerconv_best_hyperparameters.json
        model = model_class(in_channels, edge_dim=4, hidden_channels=64, num_heads=8, dropout=0.1, num_classes=num_classes)
    elif model_name == 'GINE':
        # From gine_best_hyperparameters.json
        model = model_class(in_channels, edge_dim=4, hidden_channels=256, dropout=0.3, num_classes=num_classes)
    else:
        model = model_class(in_channels, num_classes=num_classes)
    
    model = model.to(device)
    
    # Load weights if checkpoint exists
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"  ✓ Loaded {model_name} from {model_path}")
    else:
        print(f"  ⚠ Warning: {model_path} not found, using untrained model")
    
    return model


@torch.no_grad()
def get_predictions(model, data_list, device, use_edge_attr=False):
    """Get predictions for all scenarios."""
    model.eval()
    
    predictions = []
    ground_truth = []
    scenario_ids = []
    node_indices = []
    
    loader = DataLoader(data_list, batch_size=1, shuffle=False)
    
    for data in tqdm(loader, desc="Getting predictions"):
        data = data.to(device)
        
        # Forward pass
        if use_edge_attr and hasattr(data, 'edge_attr') and data.edge_attr is not None:
            out = model(data.x, data.edge_index, data.edge_attr)
        else:
            out = model(data.x, data.edge_index)
        
        # Get predictions for training nodes only
        valid_mask = data.y[data.train_mask] != -1
        
        if valid_mask.sum() > 0:
            pred = out[data.train_mask][valid_mask].argmax(dim=1)
            true = data.y[data.train_mask][valid_mask]
            
            predictions.extend(pred.cpu().numpy())
            ground_truth.extend(true.cpu().numpy())
            scenario_ids.extend([data.scenario_id.item()] * valid_mask.sum().item())
            
            # Get node indices
            train_node_indices = torch.where(data.train_mask)[0][valid_mask]
            node_indices.extend(train_node_indices.cpu().numpy())
    
    return np.array(predictions), np.array(ground_truth), np.array(scenario_ids), np.array(node_indices)


def analyze_misclassifications(all_predictions, all_ground_truth, all_scenario_ids, model_names):
    """Analyze which scenarios are consistently misclassified across models."""
    print("\n" + "="*70)
    print("ANALYZING CONSISTENTLY MISCLASSIFIED SCENARIOS")
    print("="*70)
    
    # Track misclassifications per scenario
    scenario_errors = defaultdict(lambda: {'total_predictions': 0, 'errors': 0, 'models_wrong': []})
    
    for model_name in model_names:
        preds = all_predictions[model_name]
        truth = all_ground_truth[model_name]
        scenario_ids = all_scenario_ids[model_name]
        
        for pred, true, sid in zip(preds, truth, scenario_ids):
            scenario_errors[sid]['total_predictions'] += 1
            if pred != true:
                scenario_errors[sid]['errors'] += 1
                if model_name not in scenario_errors[sid]['models_wrong']:
                    scenario_errors[sid]['models_wrong'].append(model_name)
    
    # Calculate error rates
    scenario_error_rates = []
    for sid, stats in scenario_errors.items():
        error_rate = stats['errors'] / stats['total_predictions']
        num_models_wrong = len(stats['models_wrong'])
        scenario_error_rates.append({
            'scenario_id': sid,
            'error_rate': error_rate,
            'num_models_wrong': num_models_wrong,
            'models_wrong': ', '.join(stats['models_wrong'])
        })
    
    df_errors = pd.DataFrame(scenario_error_rates)
    df_errors = df_errors.sort_values('error_rate', ascending=False)
    
    print(f"\nTop 10 Most Difficult Scenarios (Highest Error Rate):")
    print(df_errors.head(10).to_string(index=False))
    
    # Scenarios where all models fail
    all_models_wrong = df_errors[df_errors['num_models_wrong'] == len(model_names)]
    print(f"\n\nScenarios where ALL {len(model_names)} models failed:")
    print(f"  Count: {len(all_models_wrong)}")
    if len(all_models_wrong) > 0:
        print(all_models_wrong.head(10).to_string(index=False))
    
    # Save results
    df_errors.to_csv('scenario_error_analysis.csv', index=False)
    print(f"\n✓ Saved to: scenario_error_analysis.csv")
    
    return df_errors


def analyze_confusion_patterns(all_predictions, all_ground_truth, model_names, class_names=['Failed', 'Degraded', 'Normal']):
    """Analyze confusion patterns for each model."""
    print("\n" + "="*70)
    print("ANALYZING CONFUSION PATTERNS")
    print("="*70)
    
    from sklearn.metrics import confusion_matrix
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, model_name in enumerate(model_names):
        preds = all_predictions[model_name]
        truth = all_ground_truth[model_name]
        
        cm = confusion_matrix(truth, preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='YlOrRd', 
                   xticklabels=class_names, yticklabels=class_names, 
                   ax=axes[idx], cbar_kws={'label': 'Proportion'})
        axes[idx].set_title(f'{model_name} Confusion Matrix', fontsize=14, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=12)
        axes[idx].set_xlabel('Predicted Label', fontsize=12)
        
        # Print most common misclassifications
        print(f"\n{model_name} - Most Common Misclassifications:")
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i != j and cm[i, j] > 0:
                    print(f"  {class_names[i]} → {class_names[j]}: {cm[i, j]} ({cm_normalized[i, j]:.1%})")
    
    plt.tight_layout()
    plt.savefig('confusion_matrices_all_models.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved confusion matrices to: confusion_matrices_all_models.png")
    plt.close()


def analyze_by_node_features(data_list, all_predictions, all_ground_truth, all_scenario_ids, all_node_indices, model_name='GINE'):
    """Analyze errors by node features (capacity, reliability, etc.)."""
    print("\n" + "="*70)
    print(f"ANALYZING ERRORS BY NODE FEATURES ({model_name})")
    print("="*70)
    
    preds = all_predictions[model_name]
    truth = all_ground_truth[model_name]
    node_idx = all_node_indices[model_name]
    scenario_ids = all_scenario_ids[model_name]
    
    # Extract node features for misclassified nodes
    misclassified_features = []
    correct_features = []
    
    for pred, true, sid, nidx in zip(preds, truth, scenario_ids, node_idx):
        data = data_list[sid]
        node_features = data.x[nidx].cpu().numpy()
        
        if pred != true:
            misclassified_features.append(node_features)
        else:
            correct_features.append(node_features)
    
    misclassified_features = np.array(misclassified_features)
    correct_features = np.array(correct_features)
    
    # Feature names (adjust based on your actual features)
    feature_names = ['Capacity', 'Reliability', 'Lead Time', 'Cost', 'Degree', 'Betweenness']
    
    # Compare feature distributions
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, feature_name in enumerate(feature_names):
        if idx < misclassified_features.shape[1]:
            axes[idx].hist(correct_features[:, idx], bins=30, alpha=0.5, label='Correct', color='green', density=True)
            axes[idx].hist(misclassified_features[:, idx], bins=30, alpha=0.5, label='Misclassified', color='red', density=True)
            axes[idx].set_title(f'{feature_name} Distribution', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel(feature_name, fontsize=10)
            axes[idx].set_ylabel('Density', fontsize=10)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower()}_error_by_features.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved feature analysis to: {model_name.lower()}_error_by_features.png")
    plt.close()
    
    # Statistical comparison
    print(f"\nFeature Statistics Comparison:")
    print(f"{'Feature':<20} {'Correct (Mean±Std)':<25} {'Misclassified (Mean±Std)':<25} {'Difference'}")
    print("-" * 90)
    for idx, feature_name in enumerate(feature_names):
        if idx < misclassified_features.shape[1]:
            correct_mean = correct_features[:, idx].mean()
            correct_std = correct_features[:, idx].std()
            miscl_mean = misclassified_features[:, idx].mean()
            miscl_std = misclassified_features[:, idx].std()
            diff = miscl_mean - correct_mean
            print(f"{feature_name:<20} {correct_mean:>6.3f} ± {correct_std:<6.3f}      {miscl_mean:>6.3f} ± {miscl_std:<6.3f}      {diff:>+7.3f}")


def analyze_error_by_class(all_predictions, all_ground_truth, model_names, class_names=['Failed', 'Degraded', 'Normal']):
    """Analyze which classes are hardest to predict."""
    print("\n" + "="*70)
    print("ANALYZING ERROR RATES BY CLASS")
    print("="*70)
    
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    results = []
    for model_name in model_names:
        preds = all_predictions[model_name]
        truth = all_ground_truth[model_name]
        
        precision = precision_score(truth, preds, average=None, zero_division=0)
        recall = recall_score(truth, preds, average=None, zero_division=0)
        f1 = f1_score(truth, preds, average=None, zero_division=0)
        
        for class_idx, class_name in enumerate(class_names):
            results.append({
                'Model': model_name,
                'Class': class_name,
                'Precision': precision[class_idx],
                'Recall': recall[class_idx],
                'F1 Score': f1[class_idx]
            })
    
    df_class_performance = pd.DataFrame(results)
    
    # Pivot for better visualization
    for metric in ['Precision', 'Recall', 'F1 Score']:
        print(f"\n{metric} by Class:")
        pivot = df_class_performance.pivot(index='Model', columns='Class', values=metric)
        print(pivot.to_string())
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, metric in enumerate(['Precision', 'Recall', 'F1 Score']):
        pivot = df_class_performance.pivot(index='Model', columns='Class', values=metric)
        pivot.plot(kind='bar', ax=axes[idx], rot=45)
        axes[idx].set_title(f'{metric} by Class', fontsize=14, fontweight='bold')
        axes[idx].set_ylabel(metric, fontsize=12)
        axes[idx].set_xlabel('Model', fontsize=12)
        axes[idx].legend(title='Class', fontsize=10)
        axes[idx].grid(True, alpha=0.3, axis='y')
        axes[idx].set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig('per_class_performance.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved per-class performance to: per_class_performance.png")
    plt.close()
    
    df_class_performance.to_csv('per_class_performance.csv', index=False)
    print(f"✓ Saved to: per_class_performance.csv")
    
    return df_class_performance


def main():
    """Main error analysis pipeline."""
    print("="*70)
    print("ERROR ANALYSIS FOR GNN MODELS")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load data
    data_list = load_scenarios('scenario_graphs_edge_disruptions', max_scenarios=2000)
    in_channels = data_list[0].x.shape[1]
    num_classes = 3
    
    # Models to analyze
    models_config = {
        'GAT': (GATModel, 'best_gat_model.pt', False),
        'GCN': (GCNModel, 'best_gcn_model.pt', False),
        'GraphSAGE': (GraphSAGEModel, 'best_graphsage_model.pt', False),
        'GIN': (GINModel, 'best_gin_model.pt', False),
        'TransformerConv': (TransformerConvModel, 'best_transformerconv_model.pt', True),
        'GINE': (GINEModel, 'best_gine_model.pt', True)
    }
    
    # Get predictions from all models
    all_predictions = {}
    all_ground_truth = {}
    all_scenario_ids = {}
    all_node_indices = {}
    
    for model_name, (model_class, model_path, use_edge_attr) in models_config.items():
        print(f"\n{'='*70}")
        print(f"Processing {model_name}")
        print(f"{'='*70}")
        
        model = load_trained_model(model_class, model_name, model_path, in_channels, num_classes, device)
        preds, truth, scenario_ids, node_idx = get_predictions(model, data_list, device, use_edge_attr)
        
        all_predictions[model_name] = preds
        all_ground_truth[model_name] = truth
        all_scenario_ids[model_name] = scenario_ids
        all_node_indices[model_name] = node_idx
        
        accuracy = (preds == truth).mean()
        print(f"  Accuracy: {accuracy:.4f}")
    
    # Perform analyses
    model_names = list(models_config.keys())
    
    # 1. Consistently misclassified scenarios
    df_errors = analyze_misclassifications(all_predictions, all_ground_truth, all_scenario_ids, model_names)
    
    # 2. Confusion patterns
    analyze_confusion_patterns(all_predictions, all_ground_truth, model_names)
    
    # 3. Per-class performance
    df_class_perf = analyze_error_by_class(all_predictions, all_ground_truth, model_names)
    
    # 4. Error analysis by node features (for best model)
    analyze_by_node_features(data_list, all_predictions, all_ground_truth, all_scenario_ids, all_node_indices, model_name='GINE')
    
    print("\n" + "="*70)
    print("✅ ERROR ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. scenario_error_analysis.csv - Scenarios ranked by error rate")
    print("  2. confusion_matrices_all_models.png - Confusion matrices for all models")
    print("  3. per_class_performance.png - Performance breakdown by class")
    print("  4. per_class_performance.csv - Detailed per-class metrics")
    print("  5. gine_error_by_features.png - Feature distribution analysis")


if __name__ == "__main__":
    main()
