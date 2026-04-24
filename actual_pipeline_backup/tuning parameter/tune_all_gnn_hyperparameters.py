"""
Unified GNN Hyperparameter Tuning for Supply Chain Disruption Scenarios

This script performs comprehensive hyperparameter tuning for multiple GNN architectures:
- GAT (Graph Attention Network)
- GCN (Graph Convolutional Network)
- GraphSAGE
- GIN (Graph Isomorphism Network)

Uses Optuna for efficient hyperparameter search with early stopping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, GINConv, TransformerConv, GINEConv
from torch_geometric.loader import DataLoader
import numpy as np
import os
import json
from tqdm import tqdm
import optuna
from optuna.trial import TrialState
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, num_layers=3, heads=4, dropout=0.3):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(num_features, hidden_channels, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
        if num_layers > 1:
            self.convs.append(GATConv(hidden_channels * heads, num_classes, heads=1, concat=False, dropout=dropout))
    
    def forward(self, x, edge_index, edge_attr=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, num_layers=3, dropout=0.3):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_channels))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.fc = torch.nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x, edge_index, edge_attr=None):
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, num_layers=3, dropout=0.3):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(SAGEConv(num_features, hidden_channels))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.fc = torch.nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x, edge_index, edge_attr=None):
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return x


class GIN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, num_layers=3, dropout=0.3):
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        # First layer
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(num_features, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        self.convs.append(GINConv(nn1))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            nn_hidden = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GINConv(nn_hidden))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        # Last layer
        if num_layers > 1:
            nn_last = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GINConv(nn_last))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        self.fc = torch.nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x, edge_index, edge_attr=None):
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return x


class TransformerConvModel(torch.nn.Module):
    """Edge-aware TransformerConv model."""
    def __init__(self, num_features, hidden_channels, num_classes, num_layers=3, heads=4, dropout=0.3, edge_dim=4):
        super(TransformerConvModel, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(TransformerConv(num_features, hidden_channels, heads=heads, edge_dim=edge_dim, concat=True, beta=True))
        self.lns.append(nn.LayerNorm(hidden_channels * heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=edge_dim, concat=True, beta=True))
            self.lns.append(nn.LayerNorm(hidden_channels * heads))
        
        # Last layer
        if num_layers > 1:
            self.convs.append(TransformerConv(hidden_channels * heads, hidden_channels, heads=1, edge_dim=edge_dim, concat=False, beta=True))
            self.lns.append(nn.LayerNorm(hidden_channels))
        
        self.fc = nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x, edge_index, edge_attr=None):
        for i, (conv, ln) in enumerate(zip(self.convs, self.lns)):
            x = conv(x, edge_index, edge_attr)
            x = ln(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return x


class GINE(torch.nn.Module):
    """Edge-aware GINE model."""
    def __init__(self, num_features, hidden_channels, num_classes, num_layers=3, dropout=0.3, edge_dim=4):
        super(GINE, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        # First layer
        nn1 = nn.Sequential(
            nn.Linear(num_features, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.convs.append(GINEConv(nn1, edge_dim=edge_dim))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            nn_hidden = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GINEConv(nn_hidden, edge_dim=edge_dim))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Last layer
        if num_layers > 1:
            nn_last = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GINEConv(nn_last, edge_dim=edge_dim))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        self.fc = nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x, edge_index, edge_attr=None):
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return x


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_scenarios(scenario_dir='scenario_graphs_edge_disruptions', max_scenarios=None):
    """Load scenario data for training."""
    print(f"\n📂 Loading scenarios from {scenario_dir}...")
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
        mask = (data.y != -1) & data.train_mask
        if mask.sum() > 0:
            pred = out[mask].argmax(dim=1)
            total_correct += (pred == data.y[mask]).sum().item()
            total_samples += mask.sum().item()
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return accuracy


def objective(trial, model_class, train_data, val_data, device, num_features, num_classes, model_name, edge_dim=4):
    """Optuna objective function for hyperparameter tuning."""
    # Suggest hyperparameters
    hidden_channels = trial.suggest_categorical('hidden_channels', [32, 64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 2, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    
    # Model-specific parameters
    if model_name == 'GAT':
        heads = trial.suggest_categorical('heads', [2, 4, 8])
        model = model_class(num_features, hidden_channels, num_classes, num_layers, heads, dropout).to(device)
    elif model_name in ['TransformerConv', 'GINE']:
        # Edge-aware models
        if model_name == 'TransformerConv':
            heads = trial.suggest_categorical('heads', [2, 4, 8])
            model = model_class(num_features, hidden_channels, num_classes, num_layers, heads, dropout, edge_dim).to(device)
        else:  # GINE
            model = model_class(num_features, hidden_channels, num_classes, num_layers, dropout, edge_dim).to(device)
    else:
        model = model_class(num_features, hidden_channels, num_classes, num_layers, dropout).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    
    # Training loop with early stopping
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    for epoch in range(100):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    return best_val_acc


def tune_model(model_name, model_class, train_data, val_data, num_features, num_classes, n_trials=50):
    """Tune hyperparameters for a specific model."""
    print("\n" + "="*70)
    print(f"TUNING {model_name}")
    print("="*70)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Number of trials: {n_trials}")
    
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    study.optimize(
        lambda trial: objective(trial, model_class, train_data, val_data, device, num_features, num_classes, model_name),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print(f"\n📊 Best Trial for {model_name}:")
    trial = study.best_trial
    print(f"  Val Accuracy: {trial.value:.4f}")
    print(f"  Hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    best_params = trial.params
    best_params['val_accuracy'] = trial.value
    return best_params, study


def main():
    """Main hyperparameter tuning pipeline."""
    print("="*70)
    print("UNIFIED GNN HYPERPARAMETER TUNING")
    print("="*70)
    
    # Load data
    data_list = load_scenarios('scenario_graphs_edge_disruptions', max_scenarios=2000)
    train_data, val_data, test_data = split_data(data_list)
    
    # Get data dimensions
    num_features = train_data[0].x.shape[1]
    all_labels = []
    for data in train_data:
        labels = data.y[data.y != -1]
        if len(labels) > 0:
            all_labels.extend(labels.tolist())
    num_classes = len(set(all_labels))
    if num_classes == 1:
        num_classes = 3
    
    print(f"\nInput features: {num_features}")
    print(f"Output classes: {num_classes}")
    
    # Check if edge features are available
    has_edge_features = hasattr(train_data[0], 'edge_attr') and train_data[0].edge_attr is not None
    edge_dim = train_data[0].edge_attr.shape[1] if has_edge_features else 0
    
    print(f"Edge features available: {has_edge_features}")
    if has_edge_features:
        print(f"Edge feature dimension: {edge_dim}")
    
    # Define models to tune
    models_to_tune = {
        'GraphSAGE': GraphSAGE,
        'GIN': GIN
    }
    
    # Add edge-aware models if edge features are available
    if has_edge_features:
        models_to_tune['TransformerConv'] = TransformerConvModel
        models_to_tune['GINE'] = GINE
        print("\n✅ Added edge-aware models: TransformerConv, GINE")
    else:
        print("\n⚠️  No edge features found. Skipping edge-aware models.")
    
    all_results = {}
    
    # Tune each model
    for model_name, model_class in models_to_tune.items():
        best_params, study = tune_model(model_name, model_class, train_data, val_data, num_features, num_classes, n_trials=50)
        all_results[model_name] = best_params
        
        # Save individual results
        filename = f'{model_name.lower()}_best_hyperparameters.json'
        with open(filename, 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"✓ Saved to: {filename}")
    
    # Save combined results
    with open('all_gnn_best_hyperparameters.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ ALL TUNING COMPLETE!")
    print("="*70)
    print("\n📊 Summary:")
    for model_name, params in all_results.items():
        print(f"\n{model_name}:")
        print(f"  Hidden Channels: {params['hidden_channels']}")
        print(f"  Num Layers: {params['num_layers']}")
        print(f"  Dropout: {params['dropout']}")
        print(f"  Learning Rate: {params['learning_rate']:.6f}")
        print(f"  Weight Decay: {params['weight_decay']:.6f}")
        if 'heads' in params:
            print(f"  Attention Heads: {params['heads']}")
        print(f"  Val Accuracy: {params['val_accuracy']:.4f}")


if __name__ == "__main__":
    main()
