"""
ML Baseline Training with K-Fold Cross-Validation for Supply Chain Resilience

This script performs k-fold cross-validation for traditional ML models
to compare against GNN models.

Models:
- Random Forest
- XGBoost
- Gradient Boosting
- Logistic Regression
- Support Vector Machine (SVM)

Features:
- K-fold cross-validation (default: 5 folds)
- Mean and standard deviation of metrics across folds
- Detailed per-fold results
- Comparison plots
"""

import torch
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def load_scenarios(scenario_dir='scenario_graphs_edge_disruptions', max_scenarios=None):
    """Load scenario data."""
    print(f"\n📂 Loading scenarios from {scenario_dir}...")
    metadata_path = os.path.join(scenario_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    num_scenarios = metadata['num_scenarios']
    if max_scenarios:
        num_scenarios = min(num_scenarios, max_scenarios)
    
    data_list = []
    for i in tqdm(range(num_scenarios), desc="Loading"):
        scenario_path = os.path.join(scenario_dir, f'scenario_{i:05d}.pt')
        data = torch.load(scenario_path, weights_only=False)
        data_list.append(data)
    
    print(f"  ✓ Loaded {len(data_list)} scenarios")
    return data_list


def extract_features_labels(data_list):
    """Extract node features and labels from graph data."""
    X_list = []
    y_list = []
    
    for data in data_list:
        # Get node features
        features = data.x.numpy()
        labels = data.y.numpy()
        
        # Filter out unlabeled nodes (-1)
        mask = labels != -1
        X_list.append(features[mask])
        y_list.append(labels[mask])
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    return X, y


def cross_validate_ml_model(model, model_name, X, y, n_folds=5):
    """Perform k-fold cross-validation for ML model."""
    print(f"\n{'='*70}")
    print(f"K-FOLD CROSS-VALIDATION: {model_name}")
    print(f"{'='*70}")
    print(f"Number of folds: {n_folds}")
    print(f"Total samples: {len(X)}")
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        print(f"\n--- Fold {fold}/{n_folds} ---")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        num_classes = len(np.unique(y))
        average_method = 'binary' if num_classes == 2 else 'weighted'
        precision = precision_score(y_val, y_pred, average=average_method, zero_division=0)
        recall = recall_score(y_val, y_pred, average=average_method, zero_division=0)
        f1 = f1_score(y_val, y_pred, average=average_method, zero_division=0)
        
        print(f"Fold {fold} Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        
        fold_results.append({
            'fold': fold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    # Calculate statistics
    df_folds = pd.DataFrame(fold_results)
    
    print(f"\n{'='*70}")
    print(f"CROSS-VALIDATION SUMMARY: {model_name}")
    print(f"{'='*70}")
    print(f"\nMean ± Std:")
    print(f"  Accuracy:  {df_folds['accuracy'].mean():.4f} ± {df_folds['accuracy'].std():.4f}")
    print(f"  Precision: {df_folds['precision'].mean():.4f} ± {df_folds['precision'].std():.4f}")
    print(f"  Recall:    {df_folds['recall'].mean():.4f} ± {df_folds['recall'].std():.4f}")
    print(f"  F1 Score:  {df_folds['f1'].mean():.4f} ± {df_folds['f1'].std():.4f}")
    
    return df_folds


def main():
    """Main cross-validation pipeline."""
    print("="*70)
    print("ML BASELINE CROSS-VALIDATION FOR SUPPLY CHAIN RESILIENCE")
    print("="*70)
    
    # Load data
    data_list = load_scenarios('scenario_graphs_edge_disruptions', max_scenarios=2000)
    
    # Extract features and labels
    print("\nExtracting features and labels...")
    X, y = extract_features_labels(data_list)
    print(f"  ✓ Features shape: {X.shape}")
    print(f"  ✓ Labels shape: {y.shape}")
    
    # Calculate class weights
    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    class_weight_values = compute_class_weight('balanced', classes=unique_classes, y=y)
    class_weight_dict = {i: w for i, w in zip(unique_classes, class_weight_values)}
    
    print(f"\n  ✓ Number of classes: {num_classes}")
    print(f"  ✓ Class distribution:")
    for cls in unique_classes:
        count = np.sum(y == cls)
        print(f"    Class {cls}: {count} samples ({count/len(y)*100:.1f}%)")
    
    # Define ML models with optimized hyperparameters from tuning
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=50,
            max_depth=12,
            min_samples_split=18,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight=class_weight_dict,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.018385893144753068,
            subsample=0.7845385881162428,
            colsample_bytree=0.7187498404766806,
            gamma=0.8011045937645115,
            min_child_weight=2,
            reg_alpha=8.229172949635404e-06,
            reg_lambda=1.1118221168253574e-08,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.13053914405709924,
            subsample=0.9500995267785087,
            min_samples_split=12,
            min_samples_leaf=10,
            max_features=None,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            C=0.024067888361003098,
            penalty='l2',
            solver='saga',
            max_iter=1000,
            class_weight=class_weight_dict,
            random_state=42,
            n_jobs=-1
        ),
        'SVM': SVC(
            C=10.388291525506107,
            kernel='poly',
            gamma='scale',
            degree=5,
            class_weight=class_weight_dict,
            random_state=42
        )
    }
    
    all_results = {}
    
    # Cross-validate each model
    for model_name, model in models.items():
        df_folds = cross_validate_ml_model(model, model_name, X, y, n_folds=5)
        all_results[model_name] = df_folds
        
        # Save fold results
        filename = f'{model_name.lower().replace(" ", "_")}_cv_results.csv'
        df_folds.to_csv(filename, index=False)
        print(f"\n✓ Saved to: {filename}")
    
    # Create comparison plot
    print("\n" + "="*70)
    print("CREATING COMPARISON PLOTS")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # Prepare data for plotting
        plot_data = []
        for model_name, df_folds in all_results.items():
            for _, row in df_folds.iterrows():
                plot_data.append({
                    'Model': model_name,
                    'Metric': metric.capitalize(),
                    'Value': row[metric]
                })
        
        df_plot = pd.DataFrame(plot_data)
        
        # Box plot
        sns.boxplot(x='Model', y='Value', data=df_plot, ax=ax)
        ax.set_title(f'{metric.capitalize()} Across Folds', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)
        
        # Add mean markers
        for i, model_name in enumerate(all_results.keys()):
            mean_val = all_results[model_name][metric].mean()
            ax.plot(i, mean_val, 'r*', markersize=15, label='Mean' if i == 0 else '')
        
        if idx == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig('ml_cross_validation_results.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: ml_cross_validation_results.png")
    
    # Create summary table
    print("\n" + "="*70)
    print("FINAL COMPARISON TABLE")
    print("="*70)
    
    summary_data = []
    for model_name, df_folds in all_results.items():
        summary_data.append({
            'Model': model_name,
            'Accuracy': f"{df_folds['accuracy'].mean():.4f} ± {df_folds['accuracy'].std():.4f}",
            'Precision': f"{df_folds['precision'].mean():.4f} ± {df_folds['precision'].std():.4f}",
            'Recall': f"{df_folds['recall'].mean():.4f} ± {df_folds['recall'].std():.4f}",
            'F1 Score': f"{df_folds['f1'].mean():.4f} ± {df_folds['f1'].std():.4f}"
        })
    
    df_summary = pd.DataFrame(summary_data)
    print("\n" + df_summary.to_string(index=False))
    
    df_summary.to_csv('ml_cv_summary.csv', index=False)
    print(f"\n✓ Saved summary: ml_cv_summary.csv")
    
    # Rank models by F1 score
    print("\n" + "="*70)
    print("MODEL RANKING (by F1 Score)")
    print("="*70)
    
    ranking_data = []
    for model_name, df_folds in all_results.items():
        ranking_data.append({
            'Model': model_name,
            'Mean F1': df_folds['f1'].mean(),
            'Std F1': df_folds['f1'].std()
        })
    
    df_ranking = pd.DataFrame(ranking_data).sort_values('Mean F1', ascending=False)
    print("\n" + df_ranking.to_string(index=False))
    
    print("\n" + "="*70)
    print("✅ CROSS-VALIDATION COMPLETE!")
    print("="*70)
    
    print("\n📁 Output Files:")
    print("  - random_forest_cv_results.csv")
    print("  - xgboost_cv_results.csv")
    print("  - gradient_boosting_cv_results.csv")
    print("  - logistic_regression_cv_results.csv")
    print("  - svm_cv_results.csv")
    print("  - ml_cv_summary.csv")
    print("  - ml_cross_validation_results.png")


if __name__ == "__main__":
    main()
