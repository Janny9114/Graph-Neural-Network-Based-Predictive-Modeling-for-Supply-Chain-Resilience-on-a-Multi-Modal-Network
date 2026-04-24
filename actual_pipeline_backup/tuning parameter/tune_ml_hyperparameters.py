"""
ML Hyperparameter Tuning for Supply Chain Disruption Prediction

This script performs comprehensive hyperparameter tuning for traditional ML models
using Optuna for efficient search.

Models:
- Random Forest
- XGBoost
- Gradient Boosting
- Logistic Regression
- Support Vector Machine (SVM)

Features:
- Optuna-based hyperparameter optimization
- 50 trials per model
- Early stopping with pruning
- Saves best hyperparameters to JSON
"""

import torch
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import optuna
from optuna.trial import TrialState
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
        features = data.x.numpy()
        labels = data.y.numpy()
        mask = labels != -1
        X_list.append(features[mask])
        y_list.append(labels[mask])
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    return X, y


# ============================================================================
# OBJECTIVE FUNCTIONS FOR EACH MODEL
# ============================================================================

def objective_random_forest(trial, X, y, class_weight_dict):
    """Optuna objective for Random Forest."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'class_weight': class_weight_dict,
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = RandomForestClassifier(**params)
    
    # Use 3-fold CV for speed
    scores = cross_val_score(model, X, y, cv=3, scoring='f1_weighted', n_jobs=-1)
    return scores.mean()


def objective_xgboost(trial, X, y):
    """Optuna objective for XGBoost."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'mlogloss'
    }
    
    model = xgb.XGBClassifier(**params)
    
    scores = cross_val_score(model, X, y, cv=3, scoring='f1_weighted', n_jobs=-1)
    return scores.mean()


def objective_logistic_regression(trial, X, y, class_weight_dict):
    """Optuna objective for Logistic Regression."""
    params = {
        'C': trial.suggest_float('C', 1e-4, 100, log=True),
        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
        'solver': 'saga',  # Supports both L1 and L2
        'max_iter': 2000,
        'class_weight': class_weight_dict,
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = LogisticRegression(**params)
    
    scores = cross_val_score(model, X, y, cv=3, scoring='f1_weighted', n_jobs=-1)
    return scores.mean()


def objective_svm(trial, X, y, class_weight_dict):
    """Optuna objective for SVM."""
    params = {
        'C': trial.suggest_float('C', 1e-2, 100, log=True),
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
        'class_weight': class_weight_dict,
        'random_state': 42
    }
    
    if params['kernel'] == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 5)
    
    model = SVC(**params)
    
    scores = cross_val_score(model, X, y, cv=3, scoring='f1_weighted', n_jobs=-1)
    return scores.mean()


def objective_gradient_boosting(trial, X, y):
    """Optuna objective for Gradient Boosting."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': 42
    }
    
    model = GradientBoostingClassifier(**params)
    
    scores = cross_val_score(model, X, y, cv=3, scoring='f1_weighted', n_jobs=-1)
    return scores.mean()


# ============================================================================
# TUNING FUNCTIONS
# ============================================================================

def tune_model(model_name, objective_func, X, y, class_weight_dict=None, n_trials=50):
    """Tune hyperparameters for a specific model."""
    print(f"\n{'='*70}")
    print(f"TUNING {model_name}")
    print(f"{'='*70}")
    print(f"Number of trials: {n_trials}")
    print(f"Samples: {len(X)}")
    
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    
    # Create objective with fixed parameters
    if class_weight_dict is not None:
        objective = lambda trial: objective_func(trial, X, y, class_weight_dict)
    else:
        objective = lambda trial: objective_func(trial, X, y)
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n📊 Best Trial for {model_name}:")
    trial = study.best_trial
    print(f"  F1 Score (CV): {trial.value:.4f}")
    print(f"  Hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    best_params = trial.params.copy()
    best_params['cv_f1_score'] = trial.value
    
    return best_params, study


def main():
    """Main hyperparameter tuning pipeline."""
    print("="*70)
    print("ML HYPERPARAMETER TUNING FOR SUPPLY CHAIN RESILIENCE")
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
    print(f"  ✓ Class weights: {class_weight_dict}")
    
    all_results = {}
    
    # ========================================================================
    # TUNE RANDOM FOREST
    # ========================================================================
    print("\n" + "="*70)
    print("1/5: RANDOM FOREST")
    print("="*70)
    
    rf_params, rf_study = tune_model(
        'Random Forest',
        objective_random_forest,
        X, y,
        class_weight_dict,
        n_trials=50
    )
    all_results['Random Forest'] = rf_params
    
    with open('random_forest_best_hyperparameters.json', 'w') as f:
        json.dump(rf_params, f, indent=2)
    print(f"✓ Saved to: random_forest_best_hyperparameters.json")
    
    # ========================================================================
    # TUNE XGBOOST
    # ========================================================================
    print("\n" + "="*70)
    print("2/5: XGBOOST")
    print("="*70)
    
    xgb_params, xgb_study = tune_model(
        'XGBoost',
        objective_xgboost,
        X, y,
        n_trials=50
    )
    all_results['XGBoost'] = xgb_params
    
    with open('xgboost_best_hyperparameters.json', 'w') as f:
        json.dump(xgb_params, f, indent=2)
    print(f"✓ Saved to: xgboost_best_hyperparameters.json")
    
    # ========================================================================
    # TUNE GRADIENT BOOSTING
    # ========================================================================
    print("\n" + "="*70)
    print("3/5: GRADIENT BOOSTING")
    print("="*70)
    
    gb_params, gb_study = tune_model(
        'Gradient Boosting',
        objective_gradient_boosting,
        X, y,
        n_trials=50
    )
    all_results['Gradient Boosting'] = gb_params
    
    with open('gradient_boosting_best_hyperparameters.json', 'w') as f:
        json.dump(gb_params, f, indent=2)
    print(f"✓ Saved to: gradient_boosting_best_hyperparameters.json")
    
    # ========================================================================
    # TUNE LOGISTIC REGRESSION
    # ========================================================================
    print("\n" + "="*70)
    print("4/5: LOGISTIC REGRESSION")
    print("="*70)
    
    lr_params, lr_study = tune_model(
        'Logistic Regression',
        objective_logistic_regression,
        X, y,
        class_weight_dict,
        n_trials=50
    )
    all_results['Logistic Regression'] = lr_params
    
    with open('logistic_regression_best_hyperparameters.json', 'w') as f:
        json.dump(lr_params, f, indent=2)
    print(f"✓ Saved to: logistic_regression_best_hyperparameters.json")
    
    # ========================================================================
    # TUNE SVM
    # ========================================================================
    print("\n" + "="*70)
    print("5/5: SUPPORT VECTOR MACHINE")
    print("="*70)
    
    svm_params, svm_study = tune_model(
        'SVM',
        objective_svm,
        X, y,
        class_weight_dict,
        n_trials=50
    )
    all_results['SVM'] = svm_params
    
    with open('svm_best_hyperparameters.json', 'w') as f:
        json.dump(svm_params, f, indent=2)
    print(f"✓ Saved to: svm_best_hyperparameters.json")
    
    # ========================================================================
    # SAVE COMBINED RESULTS
    # ========================================================================
    with open('all_ml_best_hyperparameters.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ ALL ML TUNING COMPLETE!")
    print("="*70)
    
    print("\n📊 Summary:")
    for model_name, params in all_results.items():
        print(f"\n{model_name}:")
        print(f"  CV F1 Score: {params['cv_f1_score']:.4f}")
        print(f"  Key Parameters:")
        for key, value in params.items():
            if key != 'cv_f1_score':
                print(f"    {key}: {value}")
    
    print("\n📁 Output Files:")
    print("  - random_forest_best_hyperparameters.json")
    print("  - xgboost_best_hyperparameters.json")
    print("  - gradient_boosting_best_hyperparameters.json")
    print("  - logistic_regression_best_hyperparameters.json")
    print("  - svm_best_hyperparameters.json")
    print("  - all_ml_best_hyperparameters.json")


if __name__ == "__main__":
    main()
