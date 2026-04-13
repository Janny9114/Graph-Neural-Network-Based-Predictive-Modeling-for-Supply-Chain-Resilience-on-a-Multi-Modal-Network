"""
Benchmark ML Models on Realistic Scenario Data
Compares traditional ML models with GNN on realistic disruption scenarios.

Key Challenge:
- 7 features: [capacity, cost_factor, risk_level, reliability, x, y, buffer]
- NO production_impact_pct or is_disrupted flags (hidden)
- ML models can't use graph structure to infer hidden information
- GNN should significantly outperform on complex multi-node scenarios
"""

import torch
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def load_scenario_data(scenario_dir='scenario_graphs_edge_disruptions', add_edge_features=True, exclude_buffer=False):
    """
    Load all scenario graphs and convert to flat features.
    
    NEW: Adds edge-aware features for fair ML vs GNN comparison!
    
    Args:
        scenario_dir: Directory containing scenario files
        add_edge_features: If True, add 1-hop edge aggregations (FAIR COMPARISON)
        exclude_buffer: If True, exclude buffer feature (index 6) for fair comparison
    """
    print("="*70)
    print("LOADING REALISTIC SCENARIO DATA FOR ML BENCHMARKING")
    print("="*70)
    
    # Load metadata
    metadata_path = os.path.join(scenario_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"\nMetadata:")
    print(f"  Total scenarios: {metadata['num_scenarios']}")
    #print(f"  Features per node: {metadata['num_features']}")
    print(f"  Edge-aware features: {'ENABLED ✅' if add_edge_features else 'DISABLED ❌'}")
    print(f"  Buffer excluded: {'YES ✅' if exclude_buffer else 'NO ❌'}")
    
    # Load all scenarios and flatten to tabular data
    print(f"\n📂 Loading and flattening {metadata['num_scenarios']} scenarios...")
    
    all_features = []
    all_labels = []
    all_scenario_ids = []
    
    for i in tqdm(range(metadata['num_scenarios']), desc="Processing scenarios"):
        scenario_path = os.path.join(scenario_dir, f'scenario_{i:05d}.pt')
        data = torch.load(scenario_path, weights_only=False)
        
        # Extract labeled nodes
        labeled_mask = data.train_mask
        if labeled_mask.sum() > 0:
            node_features = data.x[labeled_mask].numpy()
            labels = data.y[labeled_mask].numpy()
            
            # EXCLUDE BUFFER (index 6) if requested
            if exclude_buffer and node_features.shape[1] > 6:
                # Keep features [0:6] (capacity, cost_factor, risk_level, reliability, x, y)
                # Skip feature [6] (buffer)
                node_features = node_features[:, :6]
            
            # ADD EDGE-AWARE FEATURES (1-hop aggregations)
            if add_edge_features and hasattr(data, 'edge_attr') and data.edge_attr is not None:
                edge_features = extract_edge_features(data, labeled_mask)
                # Concatenate node features + edge features
                features = np.hstack([node_features, edge_features])
            else:
                features = node_features
            
            all_features.append(features)
            all_labels.append(labels)
            all_scenario_ids.extend([i] * len(labels))
    
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    scenario_ids = np.array(all_scenario_ids)
    
    print(f"\n✓ Loaded {len(X):,} labeled samples")
    print(f"  Feature dimensions: {X.shape[1]} features per node")
    
    # Count classes (support both binary and multi-class)
    unique_classes = np.unique(y)
    print(f"  Classes found: {unique_classes}")
    for cls in unique_classes:
        count = (y == cls).sum()
        pct = count / len(y) * 100
        class_name = {0: 'Failed', 1: 'Lightly Degraded', 2: 'Heavily Degraded', 3: 'Normal'}.get(cls, f'Class {cls}')
        print(f"    {class_name} ({cls}): {count:,} ({pct:.1f}%)")
    
    return X, y, scenario_ids, metadata


def extract_edge_features(data, node_mask):
    """
    Extract FAIR 1-hop edge aggregation features for ML baseline.
    
    FAIR COMPARISON PRINCIPLE:
    - ML gets ONLY structural/operational metrics (no disruption signals!)
    - GNN gets full graph + disruption info through message passing
    - This creates a fair test of graph reasoning vs tabular learning
    
    Edge features extracted (4 per node):
    1. in_degree: Number of incoming edges (structural connectivity)
    2. avg_lead_time: Average lead time of incoming edges (operational)
    3. total_capacity_share: Sum of capacity shares (operational volume)
    4. avg_cost: Average cost of incoming edges (operational)
    
    Returns:
        np.array of shape (num_labeled_nodes, 4)
    """
    edge_index = data.edge_index.numpy()
    edge_attr = data.edge_attr.numpy()
    
    # Get labeled node indices
    labeled_nodes = torch.where(node_mask)[0].numpy()
    
    # Initialize edge features
    edge_features = np.zeros((len(labeled_nodes), 4))
    
    # Edge feature indices (first 4 are always: lead_time, cost, capacity_share, disruption_prob)
    lead_time_idx = 0
    cost_idx = 1
    capacity_share_idx = 2
    # Note: We deliberately ignore disruption_prob, is_disrupted, disruption_severity, time_to_recovery
    
    for idx, node in enumerate(labeled_nodes):
        # Find incoming edges to this node
        incoming_mask = edge_index[1] == node
        incoming_edge_attr = edge_attr[incoming_mask]
        
        if len(incoming_edge_attr) > 0:
            # Feature 1: In-Degree (number of incoming edges)
            in_degree = len(incoming_edge_attr)
            edge_features[idx, 0] = in_degree
            
            # Feature 2: Average lead time
            avg_lead_time = incoming_edge_attr[:, lead_time_idx].mean()
            edge_features[idx, 1] = avg_lead_time
            
            # Feature 3: Total inbound capacity share
            total_capacity_share = incoming_edge_attr[:, capacity_share_idx].sum()
            edge_features[idx, 2] = total_capacity_share
            
            # Feature 4: Average cost
            avg_cost = incoming_edge_attr[:, cost_idx].mean()
            edge_features[idx, 3] = avg_cost
    
    return edge_features


def split_by_scenarios(X, y, scenario_ids, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Split data by scenarios."""
    np.random.seed(seed)
    
    unique_scenarios = np.unique(scenario_ids)
    num_scenarios = len(unique_scenarios)
    shuffled_scenarios = np.random.permutation(unique_scenarios)
    
    train_size = int(num_scenarios * train_ratio)
    val_size = int(num_scenarios * val_ratio)
    
    train_scenarios = shuffled_scenarios[:train_size]
    test_scenarios = shuffled_scenarios[train_size + val_size:]
    
    train_mask = np.isin(scenario_ids, train_scenarios)
    test_mask = np.isin(scenario_ids, test_scenarios)
    
    print(f"\nScenario-Based Split:")
    print(f"  Train: {len(train_scenarios)} scenarios, {train_mask.sum():,} samples")
    print(f"  Test: {len(test_scenarios)} scenarios, {test_mask.sum():,} samples")
    
    return train_mask, test_mask


def train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test):
    """
    Train and evaluate a single model.
    
    NEW: Supports multi-class classification (4 classes)
    Uses 'weighted' average for precision, recall, F1
    """
    print(f"\n{'='*70}")
    print(f"Training {model_name}")
    print(f"{'='*70}")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Determine if binary or multi-class
    num_classes = len(np.unique(y_train))
    average_method = 'binary' if num_classes == 2 else 'weighted'
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=average_method, zero_division=0)
    recall = recall_score(y_test, y_pred, average=average_method, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=average_method, zero_division=0)
    
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f} ({average_method} avg)")
    print(f"  Recall:    {recall:.4f} ({average_method} avg)")
    print(f"  F1 Score:  {f1:.4f} ({average_method} avg)")
    
    return {'model': model_name, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def benchmark_models(X_train, y_train, X_test, y_test):
    """Benchmark multiple ML models."""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        #'SVM (RBF)': SVC(kernel='rbf', random_state=42)
    }
    
    results = []
    for model_name, model in models.items():
        result = train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test)
        results.append(result)
    
    return results


def load_gnn_results():
    """Load GNN results from training (auto-updated from JSON file)."""
    print("\n" + "="*70)
    print("LOADING GNN RESULTS")
    print("="*70)
    
    try:
        # Load from JSON file generated by train_gnn_realistic.py
        with open('gnn_results_realistic.json', 'r') as f:
            gnn_result = json.load(f)
        
        print(f"\n  ✓ Loaded GNN results from gnn_results_realistic.json")
        print(f"\n  GNN Test Results:")
        print(f"    Accuracy:  {gnn_result['accuracy']:.4f}")
        print(f"    Precision: {gnn_result['precision']:.4f}")
        print(f"    Recall:    {gnn_result['recall']:.4f}")
        print(f"    F1 Score:  {gnn_result['f1']:.4f}")
        
        return gnn_result
    except FileNotFoundError:
        print("  ⚠ GNN results file not found. Run train_gnn_realistic.py first.")
        return None


def plot_comparison(results, gnn_result=None):
    """Plot comparison of all models."""
    if gnn_result:
        results.append(gnn_result)
    
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        df_sorted = df.sort_values(metric, ascending=False)
        bars = ax.bar(range(len(df_sorted)), df_sorted[metric])
        
        if gnn_result:
            gnn_idx = df_sorted[df_sorted['model'] == 'GNN (GAT)'].index[0]
            bars[list(df_sorted.index).index(gnn_idx)].set_color('red')
        
        ax.set_xticks(range(len(df_sorted)))
        ax.set_xticklabels(df_sorted['model'], rotation=45, ha='right')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} Comparison')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (idx, row) in enumerate(df_sorted.iterrows()):
            ax.text(i, row[metric] + 0.02, f'{row[metric]:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('ml_benchmark_comparison_realistic.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot: ml_benchmark_comparison_realistic.png")
    plt.close()


def main():
    """Main benchmarking pipeline."""
    print("="*70)
    print("ML BENCHMARKING - REALISTIC SCENARIOS")
    print("="*70)
    
    # Load data
    X, y, scenario_ids, metadata = load_scenario_data('scenario_graphs_edge_disruptions')
    
    # Split by scenarios
    train_mask, test_mask = split_by_scenarios(X, y, scenario_ids)
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    # Benchmark ML models
    ml_results = benchmark_models(X_train, y_train, X_test, y_test)
    
    # Load GNN results
    gnn_result = load_gnn_results()
    
    # Create comparison table
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    
    all_results = ml_results.copy()
    if gnn_result:
        all_results.append(gnn_result)
    
    df_results = pd.DataFrame(all_results)
    print("\n" + df_results.to_string(index=False))
    
    # Save results
    df_results.to_csv('ml_benchmark_results_realistic.csv', index=False)
    print(f"\n✓ Saved results: ml_benchmark_results_realistic.csv")
    
    # Plot comparisons
    plot_comparison(ml_results, gnn_result)
    
    print("\n" + "="*70)
    print("✅ BENCHMARKING COMPLETE!")
    print("="*70)
    
    # Key insights
    print("\n📊 Key Insights:")
    if gnn_result:
        gnn_f1 = gnn_result['f1']
        best_ml_f1 = max([r['f1'] for r in ml_results])
        best_ml_model = [r['model'] for r in ml_results if r['f1'] == best_ml_f1][0]
        
        if gnn_f1 > best_ml_f1:
            improvement = ((gnn_f1 - best_ml_f1) / best_ml_f1) * 100
            print(f"  ✓ GNN outperforms best ML model ({best_ml_model})")
            print(f"    F1 improvement: {improvement:.1f}%")
            print(f"  ✓ GNN leverages graph structure to infer hidden production_impact_pct")
            print(f"  ✓ ML models can't see cascade patterns")
        else:
            print(f"  ⚠ Best ML model ({best_ml_model}) outperforms GNN")


if __name__ == "__main__":
    main()
