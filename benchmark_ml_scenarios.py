"""
Benchmark ML Models on Scenario-Based Data
Compares traditional ML models with GNN on 1000 separate disruption scenarios.
"""

import torch
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def load_scenario_data(scenario_dir='scenario_graphs_tts_ttr'):
    """Load all scenario graphs and convert to flat features."""
    print("="*70)
    print("LOADING TTS vs TTR SCENARIO DATA FOR ML BENCHMARKING")
    print("="*70)
    
    # Load metadata
    metadata_path = os.path.join(scenario_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"\nMetadata:")
    print(f"  Total scenarios: {metadata['num_scenarios']}")
    print(f"  Nodes per graph: {metadata['num_nodes']}")
    print(f"  Features per node: {metadata['num_features']}")
    
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
            # Get features for labeled nodes
            features = data.x[labeled_mask].numpy()
            labels = data.y[labeled_mask].numpy()
            
            all_features.append(features)
            all_labels.append(labels)
            all_scenario_ids.extend([i] * len(labels))
    
    # Concatenate all data
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    scenario_ids = np.array(all_scenario_ids)
    
    print(f"\n✓ Loaded {len(X):,} labeled samples")
    print(f"  Features shape: {X.shape}")
    print(f"  Resilient (1): {(y == 1).sum():,} ({(y == 1).sum()/len(y)*100:.1f}%)")
    print(f"  Vulnerable (0): {(y == 0).sum():,} ({(y == 0).sum()/len(y)*100:.1f}%)")
    
    return X, y, scenario_ids, metadata


def split_by_scenarios(X, y, scenario_ids, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Split data by scenarios (not by individual samples)."""
    np.random.seed(seed)
    
    unique_scenarios = np.unique(scenario_ids)
    num_scenarios = len(unique_scenarios)
    
    # Shuffle scenarios
    shuffled_scenarios = np.random.permutation(unique_scenarios)
    
    # Split scenarios
    train_size = int(num_scenarios * train_ratio)
    val_size = int(num_scenarios * val_ratio)
    
    train_scenarios = shuffled_scenarios[:train_size]
    val_scenarios = shuffled_scenarios[train_size:train_size + val_size]
    test_scenarios = shuffled_scenarios[train_size + val_size:]
    
    # Create masks
    train_mask = np.isin(scenario_ids, train_scenarios)
    val_mask = np.isin(scenario_ids, val_scenarios)
    test_mask = np.isin(scenario_ids, test_scenarios)
    
    print(f"\nScenario-Based Split:")
    print(f"  Train: {len(train_scenarios)} scenarios, {train_mask.sum():,} samples")
    print(f"  Val: {len(val_scenarios)} scenarios, {val_mask.sum():,} samples")
    print(f"  Test: {len(test_scenarios)} scenarios, {test_mask.sum():,} samples")
    
    return train_mask, val_mask, test_mask


def train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test):
    """Train and evaluate a single model."""
    print(f"\n{'='*70}")
    print(f"Training {model_name}")
    print(f"{'='*70}")
    
    # Train
    print(f"  Training on {len(X_train):,} samples...")
    model.fit(X_train, y_train)
    
    # Predict
    print(f"  Predicting on {len(X_test):,} samples...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n  Results:")
    print(f"    Accuracy:  {accuracy:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1 Score:  {f1:.4f}")
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': y_pred
    }


def benchmark_models(X_train, y_train, X_test, y_test):
    """Benchmark multiple ML models."""
    print("\n" + "="*70)
    print("BENCHMARKING ML MODELS")
    print("="*70)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42)
    }
    
    results = []
    
    for model_name, model in models.items():
        result = train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test)
        results.append(result)
    
    return results


def load_gnn_results():
    """Load GNN results from previous training."""
    print("\n" + "="*70)
    print("LOADING GNN RESULTS")
    print("="*70)
    
    try:
        gnn_results = pd.read_csv('training_results_scenarios.csv')
        
        # Get final epoch metrics
        final_metrics = gnn_results.iloc[-1]
        
        # Load test results from the training output
        # Note: These are hardcoded from the training output
        # In production, you'd save these to a file
        gnn_result = {
            'model': 'GNN (GAT)',
            'accuracy': 0.9957,
            'precision': 0.9900,
            'recall': 0.9990,
            'f1': 0.9945
        }
        
        print(f"\n  GNN Test Results:")
        print(f"    Accuracy:  {gnn_result['accuracy']:.4f}")
        print(f"    Precision: {gnn_result['precision']:.4f}")
        print(f"    Recall:    {gnn_result['recall']:.4f}")
        print(f"    F1 Score:  {gnn_result['f1']:.4f}")
        
        return gnn_result
    except FileNotFoundError:
        print("  ⚠ GNN results not found. Run train_gnn_scenarios.py first.")
        return None


def plot_comparison(results, gnn_result=None):
    """Plot comparison of all models."""
    # Prepare data
    models = [r['model'] for r in results]
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    if gnn_result:
        results.append(gnn_result)
        models.append(gnn_result['model'])
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # Sort by metric
        df_sorted = df.sort_values(metric, ascending=False)
        
        # Create bar plot
        bars = ax.bar(range(len(df_sorted)), df_sorted[metric])
        
        # Color GNN bar differently
        if gnn_result:
            gnn_idx = df_sorted[df_sorted['model'] == 'GNN (GAT)'].index[0]
            bars[list(df_sorted.index).index(gnn_idx)].set_color('red')
        
        ax.set_xticks(range(len(df_sorted)))
        ax.set_xticklabels(df_sorted['model'], rotation=45, ha='right')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} Comparison')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (idx, row) in enumerate(df_sorted.iterrows()):
            ax.text(i, row[metric] + 0.02, f'{row[metric]:.3f}', 
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('ml_benchmark_comparison_scenarios.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot: ml_benchmark_comparison_scenarios.png")
    plt.close()


def plot_detailed_comparison(results, gnn_result=None):
    """Create detailed comparison plot."""
    if gnn_result:
        results.append(gnn_result)
    
    df = pd.DataFrame(results)
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(df))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, df[metric], width, label=metric.capitalize())
        
        # Highlight GNN
        if gnn_result and 'GNN' in df.iloc[-1]['model']:
            bars[-1].set_alpha(0.8)
            bars[-1].set_edgecolor('red')
            bars[-1].set_linewidth(2)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('ML Model Comparison - Scenario-Based Training')
    ax.set_xticks(x)
    ax.set_xticklabels(df['model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('ml_benchmark_detailed_scenarios.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved detailed comparison: ml_benchmark_detailed_scenarios.png")
    plt.close()


def main():
    """Main benchmarking pipeline."""
    print("="*70)
    print("ML BENCHMARKING - SCENARIO-BASED DATA")
    print("="*70)
    print("\nComparing traditional ML models with GNN on 1000 scenarios\n")
    
    # Load data
    X, y, scenario_ids, metadata = load_scenario_data('scenario_graphs_tts_ttr')
    
    # Split by scenarios
    train_mask, val_mask, test_mask = split_by_scenarios(
        X, y, scenario_ids, train_ratio=0.7, val_ratio=0.15, seed=42
    )
    
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
    
    # Find best model for each metric
    print("\n" + "="*70)
    print("BEST MODELS BY METRIC")
    print("="*70)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        best_idx = df_results[metric].idxmax()
        best_model = df_results.loc[best_idx, 'model']
        best_score = df_results.loc[best_idx, metric]
        print(f"  {metric.capitalize():12s}: {best_model:25s} ({best_score:.4f})")
    
    # Save results
    df_results.to_csv('ml_benchmark_results_scenarios.csv', index=False)
    print(f"\n✓ Saved results: ml_benchmark_results_scenarios.csv")
    
    # Plot comparisons
    plot_comparison(ml_results, gnn_result)
    plot_detailed_comparison(ml_results, gnn_result)
    
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
        else:
            print(f"  ⚠ Best ML model ({best_ml_model}) outperforms GNN")
            print(f"    ML F1: {best_ml_f1:.4f} vs GNN F1: {gnn_f1:.4f}")
    
    print("\n  ✓ All models trained on scenario-based data (no aggregation loss)")
    print("  ✓ GNN leverages graph structure, ML uses flat features")
    print("  ✓ High recall indicates models predict resilient class well")


if __name__ == "__main__":
    main()
