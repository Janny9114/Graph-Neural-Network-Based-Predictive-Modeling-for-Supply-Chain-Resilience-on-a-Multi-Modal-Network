"""
Benchmark: Traditional ML Models vs GNN for Supply Chain Resilience Prediction

Compares the following models:
1. Graph Neural Network (GAT) - baseline
2. Random Forest
3. Gradient Boosting (XGBoost)
4. Support Vector Machine (SVM)
5. Logistic Regression
6. K-Nearest Neighbors (KNN)
7. Multi-Layer Perceptron (MLP)
"""

import torch
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import time
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Will use GradientBoosting instead.")


class MLBenchmark:
    """Benchmark traditional ML models against GNN for resilience prediction."""
    
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
        self.results = {}
        self.models = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load graph data and labels, extract node features."""
        print("="*70)
        print("LOADING DATA")
        print("="*70)
        
        # Load graph
        data = torch.load('supply_chain_graph.pt', weights_only=False)
        print(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges")
        
        # Load labels
        labels_df = pd.read_csv('node_resilience_labels.csv')
        
        # Extract node features as numpy array
        X = data.x.numpy()
        y = labels_df['resilient'].values
        
        print(f"Features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Class distribution: {np.bincount(y)}")
        
        # Create train/val/test splits (same as GNN)
        torch.manual_seed(self.seed)
        indices = torch.randperm(len(y)).numpy()
        
        train_size = int(len(y) * 0.7)
        val_size = int(len(y) * 0.15)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        # Split data
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # Standardize features
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        print(f"\nTrain: {len(X_train)} samples")
        print(f"Val: {len(X_val)} samples")
        print(f"Test: {len(X_test)} samples")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def initialize_models(self):
        """Initialize all ML models."""
        print("\n" + "="*70)
        print("INITIALIZING MODELS")
        print("="*70)
        
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=self.seed,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.seed
            ),
            'SVM (RBF)': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=self.seed
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=self.seed,
                class_weight='balanced'
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            ),
            'MLP (Neural Network)': MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                max_iter=500,
                random_state=self.seed,
                early_stopping=True
            )
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.seed,
                eval_metric='logloss'
            )
        
        print(f"Initialized {len(self.models)} models:")
        for name in self.models.keys():
            print(f"  - {name}")
    
    def train_and_evaluate(self, model_name, model, X_train, y_train, X_val, y_val, X_test, y_test):
        """Train a model and evaluate on all sets."""
        print(f"\n{'='*70}")
        print(f"Training: {model_name}")
        print(f"{'='*70}")
        
        # Train
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predict
        start_time = time.time()
        y_pred_test = model.predict(X_test)
        inference_time = (time.time() - start_time) / len(X_test) * 1000  # ms per sample
        
        # Get probabilities for ROC-AUC
        if hasattr(model, 'predict_proba'):
            y_proba_test = model.predict_proba(X_test)[:, 1]
        else:
            y_proba_test = model.decision_function(X_test)
        
        # Calculate metrics
        metrics = {
            'model': model_name,
            'train_time': train_time,
            'inference_time_ms': inference_time,
            'accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test, zero_division=0),
            'recall': recall_score(y_test, y_pred_test, zero_division=0),
            'f1': f1_score(y_test, y_pred_test, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba_test)
        }
        
        # Validation metrics
        y_pred_val = model.predict(X_val)
        metrics['val_accuracy'] = accuracy_score(y_val, y_pred_val)
        metrics['val_f1'] = f1_score(y_val, y_pred_val, zero_division=0)
        
        # Training metrics
        y_pred_train = model.predict(X_train)
        metrics['train_accuracy'] = accuracy_score(y_train, y_pred_train)
        metrics['train_f1'] = f1_score(y_train, y_pred_train, zero_division=0)
        
        # Print results
        print(f"Training time: {train_time:.2f}s")
        print(f"Inference time: {inference_time:.4f}ms per sample")
        print(f"\nTest Set Performance:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        return metrics, y_pred_test, y_proba_test
    
    def run_benchmark(self):
        """Run complete benchmark."""
        print("\n" + "="*70)
        print("ML MODELS BENCHMARK FOR SUPPLY CHAIN RESILIENCE PREDICTION")
        print("="*70)
        
        # Load data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.load_data()
        
        # Initialize models
        self.initialize_models()
        
        # Train and evaluate each model
        all_results = []
        predictions = {}
        probabilities = {}
        
        for model_name, model in self.models.items():
            metrics, y_pred, y_proba = self.train_and_evaluate(
                model_name, model, X_train, y_train, X_val, y_val, X_test, y_test
            )
            all_results.append(metrics)
            predictions[model_name] = y_pred
            probabilities[model_name] = y_proba
        
        # Add GNN results from previous training
        gnn_results = self.load_gnn_results()
        if gnn_results:
            all_results.append(gnn_results)
        
        # Create results DataFrame
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('f1', ascending=False)
        
        return results_df, predictions, probabilities, y_test
    
    def load_gnn_results(self):
        """Load GNN results from previous training."""
        try:
            gnn_df = pd.read_csv('training_results.csv')
            return {
                'model': 'GNN (GAT)',
                'train_time': np.nan,  # Not recorded
                'inference_time_ms': np.nan,
                'accuracy': gnn_df['test_accuracy'].values[0],
                'precision': gnn_df['test_precision'].values[0],
                'recall': gnn_df['test_recall'].values[0],
                'f1': gnn_df['test_f1'].values[0],
                'roc_auc': np.nan,  # Not recorded
                'val_accuracy': gnn_df['val_accuracy'].values[0],
                'val_f1': gnn_df['val_f1'].values[0],
                'train_accuracy': gnn_df['train_accuracy'].values[0],
                'train_f1': gnn_df['train_f1'].values[0]
            }
        except:
            print("Warning: Could not load GNN results")
            return None
    
    def plot_comparison(self, results_df):
        """Create comprehensive comparison plots."""
        print("\n" + "="*70)
        print("GENERATING COMPARISON PLOTS")
        print("="*70)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('ML Models Benchmark: Supply Chain Resilience Prediction', 
                     fontsize=16, fontweight='bold')
        
        # 1. F1 Score Comparison
        ax = axes[0, 0]
        results_sorted = results_df.sort_values('f1', ascending=True)
        colors = ['#2ecc71' if model == 'GNN (GAT)' else '#3498db' 
                  for model in results_sorted['model']]
        ax.barh(results_sorted['model'], results_sorted['f1'], color=colors)
        ax.set_xlabel('F1 Score')
        ax.set_title('F1 Score Comparison (Test Set)')
        ax.grid(axis='x', alpha=0.3)
        
        # 2. Accuracy Comparison
        ax = axes[0, 1]
        results_sorted = results_df.sort_values('accuracy', ascending=True)
        colors = ['#2ecc71' if model == 'GNN (GAT)' else '#3498db' 
                  for model in results_sorted['model']]
        ax.barh(results_sorted['model'], results_sorted['accuracy'], color=colors)
        ax.set_xlabel('Accuracy')
        ax.set_title('Accuracy Comparison (Test Set)')
        ax.grid(axis='x', alpha=0.3)
        
        # 3. Precision vs Recall
        ax = axes[0, 2]
        for idx, row in results_df.iterrows():
            color = '#2ecc71' if row['model'] == 'GNN (GAT)' else '#3498db'
            marker = 's' if row['model'] == 'GNN (GAT)' else 'o'
            ax.scatter(row['recall'], row['precision'], 
                      s=200, color=color, marker=marker, alpha=0.7,
                      edgecolors='black', linewidth=2)
            ax.annotate(row['model'], (row['recall'], row['precision']),
                       fontsize=8, ha='center', va='bottom')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision vs Recall Trade-off')
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # 4. Training Time Comparison
        ax = axes[1, 0]
        train_time_df = results_df[results_df['train_time'].notna()].sort_values('train_time')
        ax.barh(train_time_df['model'], train_time_df['train_time'], color='#e74c3c')
        ax.set_xlabel('Training Time (seconds)')
        ax.set_title('Training Time Comparison')
        ax.grid(axis='x', alpha=0.3)
        
        # 5. All Metrics Heatmap
        ax = axes[1, 1]
        metrics_cols = ['accuracy', 'precision', 'recall', 'f1']
        heatmap_data = results_df[['model'] + metrics_cols].set_index('model')[metrics_cols]
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                   vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Score'})
        ax.set_title('All Metrics Heatmap')
        ax.set_ylabel('')
        
        # 6. ROC-AUC Comparison
        ax = axes[1, 2]
        roc_df = results_df[results_df['roc_auc'].notna()].sort_values('roc_auc', ascending=True)
        if len(roc_df) > 0:
            ax.barh(roc_df['model'], roc_df['roc_auc'], color='#9b59b6')
            ax.set_xlabel('ROC-AUC Score')
            ax.set_title('ROC-AUC Comparison')
            ax.grid(axis='x', alpha=0.3)
            ax.set_xlim(0, 1)
        else:
            ax.text(0.5, 0.5, 'ROC-AUC data not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('ROC-AUC Comparison')
        
        plt.tight_layout()
        plt.savefig('ml_benchmark_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: ml_benchmark_comparison.png")
        plt.close()
    
    def print_summary_table(self, results_df):
        """Print formatted summary table."""
        print("\n" + "="*70)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*70)
        
        # Format table
        display_cols = ['model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        display_df = results_df[display_cols].copy()
        
        print("\n" + display_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        
        # Find best model
        best_model = results_df.loc[results_df['f1'].idxmax()]
        print(f"\n{'='*70}")
        print(f"🏆 BEST MODEL: {best_model['model']}")
        print(f"{'='*70}")
        print(f"  F1 Score:  {best_model['f1']:.4f}")
        print(f"  Accuracy:  {best_model['accuracy']:.4f}")
        print(f"  Precision: {best_model['precision']:.4f}")
        print(f"  Recall:    {best_model['recall']:.4f}")
        
        # Compare to GNN
        if 'GNN (GAT)' in results_df['model'].values:
            gnn_row = results_df[results_df['model'] == 'GNN (GAT)'].iloc[0]
            print(f"\n{'='*70}")
            print(f"GNN (GAT) Performance:")
            print(f"{'='*70}")
            print(f"  F1 Score:  {gnn_row['f1']:.4f}")
            print(f"  Accuracy:  {gnn_row['accuracy']:.4f}")
            print(f"  Rank: {results_df[results_df['model'] == 'GNN (GAT)'].index[0] + 1}/{len(results_df)}")
            
            if best_model['model'] != 'GNN (GAT)':
                improvement = ((best_model['f1'] - gnn_row['f1']) / gnn_row['f1']) * 100
                print(f"\n  Best model outperforms GNN by {improvement:.2f}% in F1 score")


def main():
    """Run complete benchmark."""
    benchmark = MLBenchmark(seed=42)
    
    # Run benchmark
    results_df, predictions, probabilities, y_test = benchmark.run_benchmark()
    
    # Save results
    results_df.to_csv('ml_benchmark_results.csv', index=False)
    print("\n✓ Results saved: ml_benchmark_results.csv")
    
    # Generate plots
    benchmark.plot_comparison(results_df)
    
    # Print summary
    benchmark.print_summary_table(results_df)
    
    print("\n" + "="*70)
    print("✓ BENCHMARK COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - ml_benchmark_results.csv (detailed metrics)")
    print("  - ml_benchmark_comparison.png (visualization)")


if __name__ == "__main__":
    main()
