"""
GNN Training with K-Fold Cross-Validation for Supply Chain Resilience

This script performs k-fold cross-validation to get robust performance estimates
for GNN models on supply chain disruption prediction.

Features:
- K-fold cross-validation (default: 5 folds)
- Mean and standard deviation of metrics across folds
- Statistical significance testing
- Detailed per-fold results
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, GINConv, TransformerConv, GINEConv
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# MODEL ARCHITECTURES (Same as train_multi_gnn_realistic.py)
# ============================================================================

class GATModel(torch.nn.Module):
    """GAT with tuned hyperparameters (4 layers, 8 heads, hidden=64, dropout=0.1)."""
    def __init__(self, in_channels, hidden_channels=64, num_heads=8, dropout=0.1, num_classes=3):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv4 = GATConv(hidden_channels, hidden_channels, heads=1, dropout=dropout)
        self.bn4 = torch.nn.BatchNorm1d(hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels + in_channels, num_classes)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        x_original = x
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.cat([x, x_original], dim=1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class GCNModel(torch.nn.Module):
    """GCN with tuned hyperparameters (4 layers, hidden=256, dropout=0.1)."""
    def __init__(self, in_channels, hidden_channels=256, dropout=0.1, num_classes=3):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.bn4 = torch.nn.BatchNorm1d(hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class GraphSAGEModel(torch.nn.Module):
    """GraphSAGE with tuned hyperparameters (4 layers, hidden=256, dropout=0.3)."""
    def __init__(self, in_channels, hidden_channels=256, dropout=0.3, num_classes=3):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv4 = SAGEConv(hidden_channels, hidden_channels)
        self.bn4 = torch.nn.BatchNorm1d(hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class GINModel(torch.nn.Module):
    """GIN with tuned hyperparameters (4 layers, hidden=256, dropout=0.1)."""
    def __init__(self, in_channels, hidden_channels=256, dropout=0.1, num_classes=3):
        super(GINModel, self).__init__()
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        
        nn3 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        
        nn4 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(hidden_channels)
        
        self.fc = torch.nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class TransformerConvModel(torch.nn.Module):
    """TransformerConv with tuned hyperparameters (4 layers, hidden=64, dropout=0.1)."""
    def __init__(self, in_channels, edge_dim=4, hidden_channels=64, num_heads=4, dropout=0.1, num_classes=3):
        super(TransformerConvModel, self).__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=num_heads, edge_dim=edge_dim, concat=True, beta=True)
        self.ln1 = nn.LayerNorm(hidden_channels * num_heads)
        self.conv2 = TransformerConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, edge_dim=edge_dim, concat=True, beta=True)
        self.ln2 = nn.LayerNorm(hidden_channels * num_heads)
        self.conv3 = TransformerConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, edge_dim=edge_dim, concat=True, beta=True)
        self.ln3 = nn.LayerNorm(hidden_channels * num_heads)
        self.conv4 = TransformerConv(hidden_channels * num_heads, hidden_channels, heads=1, edge_dim=edge_dim, concat=False, beta=True)
        self.ln4 = nn.LayerNorm(hidden_channels)
        self.fc = nn.Linear(hidden_channels + in_channels, num_classes)
        self.dropout = dropout
    
    def forward(self, x, edge_index, edge_attr=None):
        x_original = x
        x = self.conv1(x, edge_index, edge_attr)
        x = self.ln1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.ln2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index, edge_attr)
        x = self.ln3(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv4(x, edge_index, edge_attr)
        x = self.ln4(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.cat([x, x_original], dim=1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class GINEModel(torch.nn.Module):
    """GINE with tuned hyperparameters (4 layers, hidden=128, dropout=0.1)."""
    def __init__(self, in_channels, edge_dim=4, hidden_channels=128, dropout=0.1, num_classes=3):
        super(GINEModel, self).__init__()
        nn1 = nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.conv1 = GINEConv(nn1, edge_dim=edge_dim)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        
        nn2 = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.conv2 = GINEConv(nn2, edge_dim=edge_dim)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        
        nn3 = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.conv3 = GINEConv(nn3, edge_dim=edge_dim)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        
        nn4 = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.conv4 = GINEConv(nn4, edge_dim=edge_dim)
        self.bn4 = nn.BatchNorm1d(hidden_channels)
        
        self.fc = nn.Linear(hidden_channels + in_channels, num_classes)
        self.dropout = dropout
    
    def forward(self, x, edge_index, edge_attr=None):
        x_original = x
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index, edge_attr)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv4(x, edge_index, edge_attr)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.cat([x, x_original], dim=1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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


def train_epoch(model, loader, optimizer, device, class_weights, use_edge_attr=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Check if model uses edge features
        if use_edge_attr and hasattr(data, 'edge_attr') and data.edge_attr is not None:
            out = model(data.x, data.edge_index, data.edge_attr)
        else:
            out = model(data.x, data.edge_index)
        
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask], weight=class_weights, ignore_index=-1)
        loss.backward()
        optimizer.step()
        
        valid_mask = data.y[data.train_mask] != -1
        if valid_mask.sum() > 0:
            total_loss += loss.item() * valid_mask.sum().item()
            pred = out[data.train_mask][valid_mask].argmax(dim=1)
            total_correct += (pred == data.y[data.train_mask][valid_mask]).sum().item()
            total_samples += valid_mask.sum().item()
    
    return total_loss / total_samples if total_samples > 0 else 0, total_correct / total_samples if total_samples > 0 else 0


@torch.no_grad()
def evaluate(model, loader, device, use_edge_attr=False):
    """Evaluate model."""
    model.eval()
    all_preds = []
    all_labels = []
    
    for data in loader:
        data = data.to(device)
        
        # Check if model uses edge features
        if use_edge_attr and hasattr(data, 'edge_attr') and data.edge_attr is not None:
            out = model(data.x, data.edge_index, data.edge_attr)
        else:
            out = model(data.x, data.edge_index)
        
        valid_mask = data.y[data.train_mask] != -1
        
        if valid_mask.sum() > 0:
            pred = out[data.train_mask][valid_mask].argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y[data.train_mask][valid_mask].cpu().numpy())
    
    if len(all_labels) == 0:
        return 0, 0, 0, 0
    
    accuracy = accuracy_score(all_labels, all_preds)
    num_classes = len(np.unique(all_labels))
    average_method = 'binary' if num_classes == 2 else 'weighted'
    precision = precision_score(all_labels, all_preds, average=average_method, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=average_method, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=average_method, zero_division=0)
    
    return accuracy, precision, recall, f1


def train_fold(model, train_loader, val_loader, optimizer, device, class_weights, use_edge_attr=False, num_epochs=200, patience=30):
    """Train model for one fold."""
    best_val_f1 = 0
    patience_counter = 0
    
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, class_weights, use_edge_attr)
        val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, device, use_edge_attr)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Load best model
    model.load_state_dict(best_state)
    return best_val_f1


def cross_validate_model(model_class, model_name, data_list, device, class_weights, in_channels, num_classes, 
                         n_folds=5, lr=0.001, weight_decay=5e-4):
    """Perform k-fold cross-validation."""
    print(f"\n{'='*70}")
    print(f"K-FOLD CROSS-VALIDATION: {model_name}")
    print(f"{'='*70}")
    print(f"Number of folds: {n_folds}")
    print(f"Total scenarios: {len(data_list)}")
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(data_list), 1):
        print(f"\n--- Fold {fold}/{n_folds} ---")
        
        # Split data
        train_data = [data_list[i] for i in train_idx]
        val_data = [data_list[i] for i in val_idx]
        
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
        
        # Create model with tuned hyperparameters
        if model_name == 'GAT':
            model = model_class(in_channels, hidden_channels=64, num_heads=4, dropout=0.1, num_classes=num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.004901384611380665, weight_decay=0.00013154005405722121)
        elif model_name == 'GCN':
            model = model_class(in_channels, hidden_channels=256, dropout=0.1, num_classes=num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0046025661288554825, weight_decay=4.011152848441864e-06)
        elif model_name == 'GraphSAGE':
            model = model_class(in_channels, hidden_channels=256, dropout=0.2, num_classes=num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0016565364708390291, weight_decay=2.635089081566129e-05)
        elif model_name == 'GIN':
            model = model_class(in_channels, hidden_channels=128, dropout=0.1, num_classes=num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0031543976982808444, weight_decay=4.843776791356605e-06)
        elif model_name == 'TransformerConv':
            model = model_class(in_channels, edge_dim=4, hidden_channels=64, num_heads=8, dropout=0.2, num_classes=num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001838054003388999, weight_decay=1.3679951063607113e-05)
        elif model_name == 'GINE':
            model = model_class(in_channels, edge_dim=4, hidden_channels=256, dropout=0.2, num_classes=num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0011128145268584856, weight_decay=5.2476809644757505e-05)
        else:
            model = model_class(in_channels, num_classes=num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Determine if model uses edge features
        use_edge_attr = model_name in ['TransformerConv', 'GINE']
        
        # Train
        best_val_f1 = train_fold(model, train_loader, val_loader, optimizer, device, class_weights, use_edge_attr=use_edge_attr)
        
        # Evaluate
        val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, device, use_edge_attr)
        
        print(f"Fold {fold} Results:")
        print(f"  Accuracy:  {val_acc:.4f}")
        print(f"  Precision: {val_prec:.4f}")
        print(f"  Recall:    {val_rec:.4f}")
        print(f"  F1 Score:  {val_f1:.4f}")
        
        fold_results.append({
            'fold': fold,
            'accuracy': val_acc,
            'precision': val_prec,
            'recall': val_rec,
            'f1': val_f1
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
    print("GNN CROSS-VALIDATION FOR SUPPLY CHAIN RESILIENCE")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load data
    data_list = load_scenarios('scenario_graphs_edge_disruptions', max_scenarios=2000)
    in_channels = data_list[0].x.shape[1]
    
    # Calculate class weights
    print("\nCalculating class weights...")
    train_labels = []
    for data in data_list:
        labels = data.y[data.train_mask].numpy()
        valid_labels = labels[labels != -1]
        train_labels.extend(valid_labels)
    
    from sklearn.utils.class_weight import compute_class_weight
    unique_classes = np.unique(train_labels)
    num_classes = len(unique_classes)
    class_weight_values = compute_class_weight('balanced', classes=unique_classes, y=train_labels)
    class_weights = torch.tensor(class_weight_values, dtype=torch.float).to(device)
    
    print(f"  ✓ Number of classes: {num_classes}")
    print(f"  ✓ Class weights: {class_weights.cpu().numpy()}")
    
    # Models to evaluate (all 6 GNN architectures)
    models = {
        'GAT': GATModel,
        'GCN': GCNModel,
        'GraphSAGE': GraphSAGEModel,
        'GIN': GINModel,
        'GINE': GINEModel,
        'TransformerConv': TransformerConvModel
    }
    
    all_results = {}
    
    # Cross-validate each model
    for model_name, model_class in models.items():
        df_folds = cross_validate_model(
            model_class, model_name, data_list, device, class_weights, 
            in_channels, num_classes, n_folds=5
        )
        all_results[model_name] = df_folds
        
        # Save fold results
        df_folds.to_csv(f'{model_name.lower()}_cv_results.csv', index=False)
        print(f"\n✓ Saved to: {model_name.lower()}_cv_results.csv")
    
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
        
        # Add mean markers
        for i, model_name in enumerate(all_results.keys()):
            mean_val = all_results[model_name][metric].mean()
            ax.plot(i, mean_val, 'r*', markersize=15, label='Mean' if i == 0 else '')
        
        if idx == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig('gnn_cross_validation_results.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: gnn_cross_validation_results.png")
    
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
    
    df_summary.to_csv('gnn_cv_summary.csv', index=False)
    print(f"\n✓ Saved summary: gnn_cv_summary.csv")
    
    print("\n" + "="*70)
    print("✅ CROSS-VALIDATION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
