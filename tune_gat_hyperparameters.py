"""
GAT Hyperparameter Tuning for Supply Chain Disruption Scenarios

This script performs comprehensive hyperparameter tuning for GAT (Graph Attention Network)
specifically optimized for supply chain disruption cascade prediction.

Key hyperparameters tuned:
1. hidden_channels: Size of hidden layers
2. num_layers: Depth of the network
3. heads: Number of attention heads
4. dropout: Dropout rate for regularization
5. learning_rate: Optimizer learning rate
6. weight_decay: L2 regularization strength

Uses Optuna for efficient hyperparameter search with early stopping.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
import numpy as np
import os
import json
from tqdm import tqdm
import optuna
from optuna.trial import TrialState
import warnings
warnings.filterwarnings('ignore')


class GAT(torch.nn.Module):
    """
    Graph Attention Network with configurable architecture.
    """
    def __init__(self, num_features, hidden_channels, num_classes, num_layers=2, heads=4, dropout=0.3):
        super(GAT, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input layer
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(num_features, hidden_channels, heads=heads, dropout=dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
        
        # Output layer (single head for final prediction)
        if num_layers > 1:
            self.convs.append(GATConv(hidden_channels * heads, num_classes, heads=1, concat=False, dropout=dropout))
        else:
            self.convs.append(GATConv(hidden_channels * heads, num_classes, heads=1, concat=False, dropout=dropout))
    
    def forward(self, x, edge_index, edge_attr=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer (no activation, no dropout)
        x = self.convs[-1](x, edge_index)
        return x


def load_scenarios(scenario_dir='scenario_graphs_edge_disruptions', max_scenarios=None):
    """Load scenario data for training."""
    print(f"\n📂 Loading scenarios from {scenario_dir}...")
    
    # Load metadata
    metadata_path = os.path.join(scenario_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    num_scenarios = metadata['num_scenarios']
    if max_scenarios:
        num_scenarios = min(num_scenarios, max_scenarios)
    
    print(f"  Loading {num_scenarios} scenarios...")
    
    data_list = []
    for i in tqdm(range(num_scenarios), desc="Loading"):
        scenario_path = os.path.join(scenario_dir, f'scenario_{i:05d}.pt')
        data = torch.load(scenario_path, weights_only=False)
        data_list.append(data)
    
    print(f"  ✓ Loaded {len(data_list)} scenarios")
    return data_list


def split_data(data_list, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Split data into train/val/test sets."""
    np.random.seed(seed)
    indices = np.random.permutation(len(data_list))
    
    train_size = int(len(data_list) * train_ratio)
    val_size = int(len(data_list) * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_data = [data_list[i] for i in train_indices]
    val_data = [data_list[i] for i in val_indices]
    test_data = [data_list[i] for i in test_indices]
    
    print(f"\n📊 Data Split:")
    print(f"  Train: {len(train_data)} scenarios")
    print(f"  Val:   {len(val_data)} scenarios")
    print(f"  Test:  {len(test_data)} scenarios")
    
    return train_data, val_data, test_data


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index, data.edge_attr)
        
        # Filter out unlabeled nodes (y == -1)
        mask = (data.y != -1) & data.train_mask
        
        if mask.sum() > 0:
            loss = F.cross_entropy(out[mask], data.y[mask])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * mask.sum().item()
            pred = out[mask].argmax(dim=1)
            total_correct += (pred == data.y[mask]).sum().item()
            total_samples += mask.sum().item()
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_correct = 0
    total_samples = 0
    
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr)
        
        # Filter out unlabeled nodes
        mask = (data.y != -1) & data.train_mask
        
        if mask.sum() > 0:
            pred = out[mask].argmax(dim=1)
            total_correct += (pred == data.y[mask]).sum().item()
            total_samples += mask.sum().item()
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return accuracy


def objective(trial, train_data, val_data, device, num_features, num_classes):
    """
    Optuna objective function for hyperparameter tuning.
    
    Hyperparameters to tune:
    - hidden_channels: [32, 64, 128, 256]
    - num_layers: [2, 3, 4]
    - heads: [2, 4, 8]
    - dropout: [0.1, 0.2, 0.3, 0.4, 0.5]
    - learning_rate: [0.0001, 0.001, 0.01]
    - weight_decay: [0.0, 0.00001, 0.0001, 0.001]
    """
    
    # Suggest hyperparameters
    hidden_channels = trial.suggest_categorical('hidden_channels', [32, 64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 2, 4)
    heads = trial.suggest_categorical('heads', [2, 4, 8])
    dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    # ✅ FIX: weight_decay must start > 0 for log=True, or use log=False
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    
    # Create model
    model = GAT(
        num_features=num_features,
        hidden_channels=hidden_channels,
        num_classes=num_classes,
        num_layers=num_layers,
        heads=heads,
        dropout=dropout
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    
    # Training loop with early stopping
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(100):  # Max 100 epochs
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        
        # Report intermediate value for pruning
        trial.report(val_acc, epoch)
        
        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return best_val_acc


def tune_hyperparameters(train_data, val_data, n_trials=50):
    """
    Run hyperparameter tuning using Optuna.
    
    Args:
        train_data: Training data
        val_data: Validation data
        n_trials: Number of trials to run
    
    Returns:
        best_params: Dictionary of best hyperparameters
        study: Optuna study object
    """
    print("\n" + "="*70)
    print("STARTING HYPERPARAMETER TUNING")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Get data dimensions
    sample_data = train_data[0]
    num_features = sample_data.x.shape[1]
    
    # ✅ FIX: Collect all unique labels across ALL scenarios
    all_labels = []
    for data in train_data:
        labels = data.y[data.y != -1]
        if len(labels) > 0:
            all_labels.extend(labels.tolist())
    
    num_classes = len(set(all_labels))
    
    # ✅ FIX: If only 1 class, force to 3 (Failed, Degraded, Normal)
    if num_classes == 1:
        print(f"⚠️ Warning: Only 1 class detected in training data!")
        print(f"   This means all labeled nodes have the same label.")
        print(f"   Forcing num_classes=3 to avoid CUDA assertion error.")
        num_classes = 3
    
    print(f"Input features: {num_features}")
    print(f"Output classes: {num_classes}")
    print(f"Number of trials: {n_trials}")
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, train_data, val_data, device, num_features, num_classes),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Print results
    print("\n" + "="*70)
    print("TUNING RESULTS")
    print("="*70)
    
    print(f"\nNumber of finished trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == TrialState.PRUNED])}")
    print(f"Number of complete trials: {len([t for t in study.trials if t.state == TrialState.COMPLETE])}")
    
    print("\n📊 Best Trial:")
    trial = study.best_trial
    print(f"  Value (Val Accuracy): {trial.value:.4f}")
    print(f"\n  Hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save results
    best_params = trial.params
    best_params['val_accuracy'] = trial.value
    
    with open('gat_best_hyperparameters.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print(f"\n✓ Saved best hyperparameters to: gat_best_hyperparameters.json")
    
    # Plot optimization history
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Optimization history
        optuna.visualization.matplotlib.plot_optimization_history(study, ax=axes[0])
        axes[0].set_title('Optimization History')
        
        # Parameter importances
        optuna.visualization.matplotlib.plot_param_importances(study, ax=axes[1])
        axes[1].set_title('Hyperparameter Importances')
        
        plt.tight_layout()
        plt.savefig('gat_tuning_results.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved tuning plots to: gat_tuning_results.png")
        plt.close()
    except Exception as e:
        print(f"⚠️ Could not generate plots: {e}")
    
    return best_params, study


def test_best_model(best_params, train_data, val_data, test_data):
    """
    Train final model with best hyperparameters and evaluate on test set.
    """
    print("\n" + "="*70)
    print("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get data dimensions
    sample_data = train_data[0]
    num_features = sample_data.x.shape[1]
    num_classes = len(torch.unique(sample_data.y[sample_data.y != -1]))
    
    # Create model with best hyperparameters
    model = GAT(
        num_features=num_features,
        hidden_channels=best_params['hidden_channels'],
        num_classes=num_classes,
        num_layers=best_params['num_layers'],
        heads=best_params['heads'],
        dropout=best_params['dropout']
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay']
    )
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    # Training loop
    best_val_acc = 0
    patience = 15
    patience_counter = 0
    
    print("\nTraining final model...")
    for epoch in range(200):  # Max 200 epochs for final model
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_gat_tuned.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load('best_gat_tuned.pt', weights_only=True))
    test_acc = evaluate(model, test_loader, device)
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"\nBest Val Accuracy:  {best_val_acc:.4f}")
    print(f"Test Accuracy:      {test_acc:.4f}")
    
    # Save final results
    final_results = {
        'best_hyperparameters': best_params,
        'val_accuracy': float(best_val_acc),
        'test_accuracy': float(test_acc)
    }
    
    with open('gat_tuning_final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n✓ Saved final results to: gat_tuning_final_results.json")
    print(f"✓ Saved best model to: best_gat_tuned.pt")
    
    return test_acc


def main():
    """Main hyperparameter tuning pipeline."""
    print("="*70)
    print("GAT HYPERPARAMETER TUNING FOR SUPPLY CHAIN DISRUPTIONS")
    print("="*70)
    
    # Load data
    data_list = load_scenarios('scenario_graphs_edge_disruptions', max_scenarios=2000)
    
    # Split data
    train_data, val_data, test_data = split_data(data_list)
    
    # Run hyperparameter tuning
    best_params, study = tune_hyperparameters(train_data, val_data, n_trials=50)
    
    # Train final model with best hyperparameters
    test_acc = test_best_model(best_params, train_data, val_data, test_data)
    
    print("\n" + "="*70)
    print("✅ HYPERPARAMETER TUNING COMPLETE!")
    print("="*70)
    print(f"\n📊 Best Configuration:")
    print(f"  Hidden Channels: {best_params['hidden_channels']}")
    print(f"  Num Layers: {best_params['num_layers']}")
    print(f"  Attention Heads: {best_params['heads']}")
    print(f"  Dropout: {best_params['dropout']}")
    print(f"  Learning Rate: {best_params['learning_rate']:.6f}")
    print(f"  Weight Decay: {best_params['weight_decay']:.6f}")
    print(f"\n  Test Accuracy: {test_acc:.4f}")
    
    print("\n📁 Output Files:")
    print("  - gat_best_hyperparameters.json")
    print("  - gat_tuning_final_results.json")
    print("  - best_gat_tuned.pt")
    print("  - gat_tuning_results.png")


if __name__ == "__main__":
    main()
