"""
Heterogeneous GNN Training for Supply Chain Resilience Prediction
Uses HeteroGNN to model different node types (Supplier, Manufacturer, Distributor, Retailer)
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, Linear
from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


class HeteroGATResiliencePredictor(torch.nn.Module):
    """
    Heterogeneous Graph Attention Network for supply chain resilience prediction.
    
    Architecture:
    - Separate processing for each node type (Supplier, Manufacturer, Distributor, Retailer)
    - 3 HeteroConv layers with GAT for each edge type
    - Multi-head attention (4 heads)
    - MLP classifier (2-layer)
    - Batch normalization and dropout
    """
    
    def __init__(self, in_channels, hidden_channels=128, num_heads=4, dropout=0.3):
        super(HeteroGATResiliencePredictor, self).__init__()
        
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        
        # Layer 1: Heterogeneous convolution
        self.conv1 = HeteroConv({
            ('supplier', 'supplies', 'manufacturer'): GATConv(in_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout),
            ('manufacturer', 'distributes', 'distributor'): GATConv(in_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout),
            ('distributor', 'sells', 'retailer'): GATConv(in_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout),
        }, aggr='sum')
        
        # Batch normalization for each node type
        self.bn1 = torch.nn.ModuleDict({
            'supplier': torch.nn.BatchNorm1d(hidden_channels),
            'manufacturer': torch.nn.BatchNorm1d(hidden_channels),
            'distributor': torch.nn.BatchNorm1d(hidden_channels),
            'retailer': torch.nn.BatchNorm1d(hidden_channels),
        })
        
        # Layer 2: Heterogeneous convolution
        self.conv2 = HeteroConv({
            ('supplier', 'supplies', 'manufacturer'): GATConv(hidden_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout),
            ('manufacturer', 'distributes', 'distributor'): GATConv(hidden_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout),
            ('distributor', 'sells', 'retailer'): GATConv(hidden_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout),
        }, aggr='sum')
        
        self.bn2 = torch.nn.ModuleDict({
            'supplier': torch.nn.BatchNorm1d(hidden_channels),
            'manufacturer': torch.nn.BatchNorm1d(hidden_channels),
            'distributor': torch.nn.BatchNorm1d(hidden_channels),
            'retailer': torch.nn.BatchNorm1d(hidden_channels),
        })
        
        # Layer 3: Heterogeneous convolution
        self.conv3 = HeteroConv({
            ('supplier', 'supplies', 'manufacturer'): GATConv(hidden_channels, hidden_channels, heads=1, dropout=dropout),
            ('manufacturer', 'distributes', 'distributor'): GATConv(hidden_channels, hidden_channels, heads=1, dropout=dropout),
            ('distributor', 'sells', 'retailer'): GATConv(hidden_channels, hidden_channels, heads=1, dropout=dropout),
        }, aggr='sum')
        
        self.bn3 = torch.nn.ModuleDict({
            'supplier': torch.nn.BatchNorm1d(hidden_channels),
            'manufacturer': torch.nn.BatchNorm1d(hidden_channels),
            'distributor': torch.nn.BatchNorm1d(hidden_channels),
            'retailer': torch.nn.BatchNorm1d(hidden_channels),
        })
        
        # MLP Classifier (2-layer as per paper)
        self.fc1 = torch.nn.ModuleDict({
            'supplier': torch.nn.Linear(hidden_channels, hidden_channels // 2),
            'manufacturer': torch.nn.Linear(hidden_channels, hidden_channels // 2),
            'distributor': torch.nn.Linear(hidden_channels, hidden_channels // 2),
            'retailer': torch.nn.Linear(hidden_channels, hidden_channels // 2),
        })
        
        self.fc2 = torch.nn.ModuleDict({
            'supplier': torch.nn.Linear(hidden_channels // 2, 2),
            'manufacturer': torch.nn.Linear(hidden_channels // 2, 2),
            'distributor': torch.nn.Linear(hidden_channels // 2, 2),
            'retailer': torch.nn.Linear(hidden_channels // 2, 2),
        })
    
    def forward(self, x_dict, edge_index_dict):
        # Layer 1
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: self.bn1[key](x) for key, x in x_dict.items()}
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) for key, x in x_dict.items()}
        
        # Layer 2
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: self.bn2[key](x) for key, x in x_dict.items()}
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) for key, x in x_dict.items()}
        
        # Layer 3
        x_dict = self.conv3(x_dict, edge_index_dict)
        x_dict = {key: self.bn3[key](x) for key, x in x_dict.items()}
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) for key, x in x_dict.items()}
        
        # MLP Classifier (2-layer)
        out_dict = {}
        for key, x in x_dict.items():
            x = self.fc1[key](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.fc2[key](x)
            out_dict[key] = F.log_softmax(x, dim=1)
        
        return out_dict


def load_heterogeneous_data(
    node_path='synthetic_nodes.csv',
    edge_path='synthetic_edges.csv',
    labels_path='node_resilience_labels_cascading.csv'
):
    """
    Load data and convert to heterogeneous graph format.
    
    Returns:
        HeteroData object with separate node types
    """
    print("Loading data for heterogeneous graph...")
    
    # Load data
    node_df = pd.read_csv(node_path)
    edge_df = pd.read_csv(edge_path)
    labels_df = pd.read_csv(labels_path)
    
    # Create HeteroData object
    data = HeteroData()
    
    # Tier mapping
    tier_names = {0: 'supplier', 1: 'manufacturer', 2: 'distributor', 3: 'retailer'}
    
    # Prepare node features for each type
    feature_columns = ['capacity', 'cost_factor', 'risk_level', 'reliability', 'x', 'y']
    
    # Add DRNL if available
    graph_data = torch.load('supply_chain_graph_cascading_drnl.pt', weights_only=False)
    node_features_with_drnl = graph_data.x  # Shape: [200, 11] (10 base + 1 DRNL)
    
    # Split nodes by tier
    for tier, tier_name in tier_names.items():
        tier_nodes = node_df[node_df['tier'] == tier]
        tier_indices = tier_nodes.index.tolist()
        
        # Extract features for this tier
        tier_features = node_features_with_drnl[tier_indices]
        data[tier_name].x = tier_features
        
        # Add labels
        tier_labels = labels_df.loc[tier_indices, 'resilient'].values
        data[tier_name].y = torch.tensor(tier_labels, dtype=torch.long)
        
        # Store original indices for mapping
        data[tier_name].original_idx = torch.tensor(tier_indices, dtype=torch.long)
        
        print(f"  {tier_name}: {len(tier_indices)} nodes, features: {tier_features.shape}")
    
    # Create edge indices for each edge type
    # Map global indices to tier-specific indices
    global_to_tier_idx = {}
    for tier, tier_name in tier_names.items():
        tier_nodes = node_df[node_df['tier'] == tier].index.tolist()
        for local_idx, global_idx in enumerate(tier_nodes):
            global_to_tier_idx[global_idx] = (tier_name, local_idx)
    
    # Process edges
    edge_types = {
        (0, 1): ('supplier', 'supplies', 'manufacturer'),
        (1, 2): ('manufacturer', 'distributes', 'distributor'),
        (2, 3): ('distributor', 'sells', 'retailer'),
    }
    
    edge_counts = {}
    for (src_tier, dst_tier), edge_type in edge_types.items():
        edge_list = []
        for _, edge in edge_df.iterrows():
            src, dst = edge['source'], edge['target']
            src_node_tier = node_df.loc[src, 'tier']
            dst_node_tier = node_df.loc[dst, 'tier']
            
            if src_node_tier == src_tier and dst_node_tier == dst_tier:
                src_tier_name, src_local_idx = global_to_tier_idx[src]
                dst_tier_name, dst_local_idx = global_to_tier_idx[dst]
                edge_list.append([src_local_idx, dst_local_idx])
        
        if edge_list:
            data[edge_type].edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_counts[edge_type] = len(edge_list)
            print(f"  {edge_type}: {len(edge_list)} edges")
        else:
            print(f"  {edge_type}: 0 edges (skipped)")
    
    # Print class distribution
    print(f"\nClass Distribution:")
    for tier_name in tier_names.values():
        if tier_name in data.node_types:
            resilient = (data[tier_name].y == 1).sum().item()
            total = len(data[tier_name].y)
            print(f"  {tier_name}: {resilient}/{total} resilient ({resilient/total*100:.1f}%)")
    
    return data, node_df


def create_train_val_test_masks(data, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Create train/val/test masks for each node type."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    tier_names = ['supplier', 'manufacturer', 'distributor', 'retailer']
    
    for tier_name in tier_names:
        if tier_name not in data.node_types:
            continue
        
        num_nodes = data[tier_name].x.shape[0]
        indices = torch.randperm(num_nodes)
        
        train_size = int(num_nodes * train_ratio)
        val_size = int(num_nodes * val_ratio)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True
        
        data[tier_name].train_mask = train_mask
        data[tier_name].val_mask = val_mask
        data[tier_name].test_mask = test_mask
        
        print(f"{tier_name}: Train={train_mask.sum()}, Val={val_mask.sum()}, Test={test_mask.sum()}")
    
    return data


def train_epoch(model, data, optimizer, class_weights_dict):
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    out_dict = model(data.x_dict, data.edge_index_dict)
    
    # Calculate loss for each node type
    total_loss = 0
    tier_names = ['supplier', 'manufacturer', 'distributor', 'retailer']
    
    for tier_name in tier_names:
        if tier_name not in data.node_types:
            continue
        
        train_mask = data[tier_name].train_mask
        if train_mask.sum() == 0:
            continue
        
        out = out_dict[tier_name][train_mask]
        y = data[tier_name].y[train_mask]
        
        # Get class weights for this tier
        class_weights = class_weights_dict.get(tier_name, None)
        if class_weights is not None:
            loss = F.nll_loss(out, y, weight=class_weights)
        else:
            loss = F.nll_loss(out, y)
        
        total_loss += loss
    
    # Backward pass
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()


@torch.no_grad()
def evaluate(model, data, mask_name='val_mask'):
    """Evaluate model."""
    model.eval()
    
    # Forward pass
    out_dict = model(data.x_dict, data.edge_index_dict)
    
    # Collect predictions and labels from all node types
    all_y_true = []
    all_y_pred = []
    
    tier_names = ['supplier', 'manufacturer', 'distributor', 'retailer']
    
    for tier_name in tier_names:
        if tier_name not in data.node_types:
            continue
        
        mask = getattr(data[tier_name], mask_name)
        if mask.sum() == 0:
            continue
        
        out = out_dict[tier_name][mask]
        y = data[tier_name].y[mask]
        
        pred = out.argmax(dim=1)
        
        all_y_true.extend(y.cpu().numpy())
        all_y_pred.extend(pred.cpu().numpy())
    
    # Calculate metrics
    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return accuracy, precision, recall, f1, y_true, y_pred


def plot_training_history(train_losses, val_f1_scores, filename='hetero_training_history.png'):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(train_losses, label='Training Loss', color='blue')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(val_f1_scores, label='Validation F1 Score', color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('Validation F1 Score Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Training history saved: {filename}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, filename='hetero_confusion_matrix.png'):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Vulnerable', 'Resilient'],
                yticklabels=['Vulnerable', 'Resilient'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Heterogeneous GNN - Test Set Confusion Matrix')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved: {filename}")
    plt.close()


def main():
    """Main training pipeline for heterogeneous GNN."""
    print("="*70)
    print("HETEROGENEOUS GNN TRAINING")
    print("="*70)
    
    # Set device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Load heterogeneous data
    data, node_df = load_heterogeneous_data()
    data = data.to(device)
    
    # Create train/val/test splits
    print("\nCreating train/val/test splits...")
    data = create_train_val_test_masks(data)
    
    # Initialize model
    print("\n" + "="*70)
    print("INITIALIZING HETEROGENEOUS GNN MODEL")
    print("="*70)
    
    in_channels = data['supplier'].x.shape[1]  # Should be 11 (10 base + 1 DRNL)
    model = HeteroGATResiliencePredictor(
        in_channels=in_channels,
        hidden_channels=128,
        num_heads=4,
        dropout=0.3
    ).to(device)
    
    print(f"Model: HeteroGATResiliencePredictor")
    print(f"  Input features: {in_channels}")
    print(f"  Hidden channels: 128")
    print(f"  GAT layers: 3 (heterogeneous)")
    print(f"  Attention heads: 4")
    print(f"  MLP classifier: 2-layer (128 → 64 → 2)")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # Calculate class weights for each node type
    print("\nCalculating class weights...")
    class_weights_dict = {}
    tier_names = ['supplier', 'manufacturer', 'distributor', 'retailer']
    
    for tier_name in tier_names:
        if tier_name not in data.node_types:
            continue
        
        train_mask = data[tier_name].train_mask
        if train_mask.sum() == 0:
            continue
        
        y_train = data[tier_name].y[train_mask]
        class_counts = torch.bincount(y_train)
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum()
        class_weights = class_weights.to(device)
        class_weights_dict[tier_name] = class_weights
        print(f"  {tier_name}: {class_weights.tolist()}")
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    num_epochs = 200
    best_val_f1 = 0
    best_epoch = 0
    patience = 30
    patience_counter = 0
    
    train_losses = []
    val_f1_scores = []
    
    for epoch in range(1, num_epochs + 1):
        # Train
        loss = train_epoch(model, data, optimizer, class_weights_dict)
        train_losses.append(loss)
        
        # Evaluate
        val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(model, data, 'val_mask')
        val_f1_scores.append(val_f1)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), 'best_hetero_gnn_model.pt')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch} (best epoch: {best_epoch})")
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_hetero_gnn_model.pt', map_location=device))
    
    # Plot training history
    plot_training_history(train_losses, val_f1_scores)
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    train_acc, train_prec, train_rec, train_f1, _, _ = evaluate(model, data, 'train_mask')
    print(f"\nTraining Set:")
    print(f"  Accuracy:  {train_acc:.4f}")
    print(f"  Precision: {train_prec:.4f}")
    print(f"  Recall:    {train_rec:.4f}")
    print(f"  F1 Score:  {train_f1:.4f}")
    
    val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(model, data, 'val_mask')
    print(f"\nValidation Set:")
    print(f"  Accuracy:  {val_acc:.4f}")
    print(f"  Precision: {val_prec:.4f}")
    print(f"  Recall:    {val_rec:.4f}")
    print(f"  F1 Score:  {val_f1:.4f}")
    
    test_acc, test_prec, test_rec, test_f1, y_true, y_pred = evaluate(model, data, 'test_mask')
    print(f"\nTest Set:")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall:    {test_rec:.4f}")
    print(f"  F1 Score:  {test_f1:.4f}")
    
    # Classification report
    print("\n" + "="*70)
    print("DETAILED CLASSIFICATION REPORT (Test Set)")
    print("="*70)
    print(classification_report(y_true, y_pred,
                                target_names=['Vulnerable', 'Resilient'],
                                digits=4))
    
    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred)
    
    # Save results
    results_df = pd.DataFrame([{
        'model': 'Heterogeneous GNN',
        'best_epoch': best_epoch,
        'test_accuracy': test_acc,
        'test_precision': test_prec,
        'test_recall': test_rec,
        'test_f1': test_f1
    }])
    
    results_df.to_csv('hetero_gnn_results.csv', index=False)
    print("\n✓ Results saved: hetero_gnn_results.csv")
    
    print("\n" + "="*70)
    print("✓ HETEROGENEOUS GNN TRAINING COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - best_hetero_gnn_model.pt (trained heterogeneous GNN)")
    print("  - hetero_training_history.png (training curves)")
    print("  - hetero_confusion_matrix.png (test predictions)")
    print("  - hetero_gnn_results.csv (performance metrics)")


if __name__ == "__main__":
    main()
