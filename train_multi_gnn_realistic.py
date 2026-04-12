"""
Multi-GNN Architecture Training for Supply Chain Resilience
Trains and compares multiple GNN architectures: GAT, GCN, GraphSAGE, GIN

Architectures:
1. GAT (Graph Attention Network) - Attention-based aggregation
2. GCN (Graph Convolutional Network) - Simple spectral convolution
3. GraphSAGE - Sampling-based aggregation
4. GIN (Graph Isomorphism Network) - Most expressive GNN
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, GINConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from tqdm import tqdm

# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class GATModel(torch.nn.Module):
    """Graph Attention Network"""
    def __init__(self, in_channels, hidden_channels=64, num_heads=4, dropout=0.3):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels, heads=1, dropout=dropout)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, 2)
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
        
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class GCNModel(torch.nn.Module):
    """Graph Convolutional Network"""
    def __init__(self, in_channels, hidden_channels=64, dropout=0.3):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, 2)
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
        
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class GraphSAGEModel(torch.nn.Module):
    """GraphSAGE with mean aggregation"""
    def __init__(self, in_channels, hidden_channels=64, dropout=0.3):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, 2)
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
        
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class GINModel(torch.nn.Module):
    """Graph Isomorphism Network"""
    def __init__(self, in_channels, hidden_channels=64, dropout=0.3):
        super(GINModel, self).__init__()
        
        # GIN uses MLPs as update functions
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
        
        self.fc = torch.nn.Linear(hidden_channels, 2)
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
        
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def load_scenario_data(scenario_dir='scenario_graphs_realistic'):
    """Load all scenario graphs from directory."""
    metadata_path = os.path.join(scenario_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    scenarios = []
    for i in tqdm(range(metadata['num_scenarios']), desc="Loading scenarios"):
        scenario_path = os.path.join(scenario_dir, f'scenario_{i:05d}.pt')
        data = torch.load(scenario_path, weights_only=False)
        scenarios.append(data)
    
    return scenarios, metadata


def split_scenarios(scenarios, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Split scenarios into train/val/test sets."""
    np.random.seed(seed)
    num_scenarios = len(scenarios)
    indices = np.random.permutation(num_scenarios)
    
    train_size = int(num_scenarios * train_ratio)
    val_size = int(num_scenarios * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_scenarios = [scenarios[i] for i in train_indices]
    val_scenarios = [scenarios[i] for i in val_indices]
    test_scenarios = [scenarios[i] for i in test_indices]
    
    return train_scenarios, val_scenarios, test_scenarios


def train_epoch(model, loader, optimizer, device, class_weights):
    """Train for one epoch with class weights."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask], weight=class_weights)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.train_mask.sum().item()
        pred = out[data.train_mask].argmax(dim=1)
        total_correct += (pred == data.y[data.train_mask]).sum().item()
        total_samples += data.train_mask.sum().item()
    
    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        
        total_loss += loss.item() * data.train_mask.sum().item()
        total_samples += data.train_mask.sum().item()
        
        pred = out[data.train_mask].argmax(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(data.y[data.train_mask].cpu().numpy())
    
    avg_loss = total_loss / total_samples
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1, all_preds, all_labels


def train_model(model, model_name, train_loader, val_loader, optimizer, device, class_weights, num_epochs=50, patience=10):
    """Train model with early stopping."""
    print(f"\n{'='*70}")
    print(f"Training {model_name}")
    print(f"{'='*70}")
    
    best_val_f1 = 0
    patience_counter = 0
    
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, class_weights)
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(model, val_loader, device)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), f'best_{model_name.lower().replace(" ", "_")}_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    print(f"Best validation F1: {best_val_f1:.4f}")
    return best_val_f1


def main():
    """Main training pipeline."""
    print("="*70)
    print("MULTI-GNN ARCHITECTURE COMPARISON")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load data
    print("\nLoading data...")
    scenarios, metadata = load_scenario_data('scenario_graphs_realistic')
    train_scenarios, val_scenarios, test_scenarios = split_scenarios(scenarios)
    
    train_loader = DataLoader(train_scenarios, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_scenarios, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_scenarios, batch_size=32, shuffle=False)
    
    in_channels = metadata['num_features']
    class_weights = torch.tensor([1.3, 1.0], dtype=torch.float).to(device)
    
    # Define models
    models = {
        'GAT': GATModel(in_channels, hidden_channels=64, num_heads=4, dropout=0.3),
        'GCN': GCNModel(in_channels, hidden_channels=64, dropout=0.3),
        'GraphSAGE': GraphSAGEModel(in_channels, hidden_channels=64, dropout=0.3),
        'GIN': GINModel(in_channels, hidden_channels=64, dropout=0.3)
    }
    
    results = []
    
    # Train each model
    for model_name, model in models.items():
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        
        # Train
        best_val_f1 = train_model(model, model_name, train_loader, val_loader, optimizer, device, class_weights)
        
        # Load best model and evaluate on test set
        model.load_state_dict(torch.load(f'best_{model_name.lower().replace(" ", "_")}_model.pt'))
        test_loss, test_acc, test_prec, test_rec, test_f1, test_preds, test_labels = evaluate(model, test_loader, device)
        
        print(f"\n{model_name} Test Results:")
        print(f"  Accuracy:  {test_acc:.4f}")
        print(f"  Precision: {test_prec:.4f}")
        print(f"  Recall:    {test_rec:.4f}")
        print(f"  F1 Score:  {test_f1:.4f}")
        
        results.append({
            'model': model_name,
            'accuracy': test_acc,
            'precision': test_prec,
            'recall': test_rec,
            'f1': test_f1
        })
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv('multi_gnn_results.csv', index=False)
    
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print("\n" + df_results.to_string(index=False))
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        df_sorted = df_results.sort_values(metric, ascending=False)
        bars = ax.bar(range(len(df_sorted)), df_sorted[metric])
        
        # Highlight best model
        bars[0].set_color('green')
        
        ax.set_xticks(range(len(df_sorted)))
        ax.set_xticklabels(df_sorted['model'], rotation=45, ha='right')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} Comparison')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (idx, row) in enumerate(df_sorted.iterrows()):
            ax.text(i, row[metric] + 0.02, f'{row[metric]:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('multi_gnn_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot: multi_gnn_comparison.png")
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
