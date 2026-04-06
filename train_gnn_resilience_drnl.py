"""
GNN Training for Supply Chain Resilience Prediction WITH DRNL LABELS
Trains a Graph Attention Network (GAT) using the DRNL-enhanced graph
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class GATResiliencePredictor(torch.nn.Module):
    """
    Graph Attention Network for predicting supply chain node resilience.
    NOW WITH DRNL STRUCTURAL LABELS!
    """
    
    def __init__(self, in_channels, hidden_channels=64, num_heads=4, dropout=0.3):
        super(GATResiliencePredictor, self).__init__()
        
        # First GAT layer: in_channels -> hidden_channels (4 heads)
        self.conv1 = GATConv(in_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        
        # Second GAT layer: hidden_channels -> hidden_channels (4 heads)
        self.conv2 = GATConv(hidden_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        
        # Third GAT layer: hidden_channels -> hidden_channels (1 head)
        self.conv3 = GATConv(hidden_channels, hidden_channels, heads=1, dropout=dropout)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        
        # Output layer: hidden_channels -> 2 classes (resilient/vulnerable)
        self.fc = torch.nn.Linear(hidden_channels, 2)
        
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 3
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def load_data():
    """Load DRNL-enhanced graph data and resilience labels."""
    print("Loading data...")
    
    # Load DRNL-enhanced graph
    data = torch.load('supply_chain_graph_drnl.pt', weights_only=False)
    print(f"  Graph loaded: {data.num_nodes} nodes, {data.num_edges} edges")
    print(f"  Node features: {data.x.shape} (includes DRNL label!)")
    print(f"  Edge features: {data.edge_attr.shape if hasattr(data, 'edge_attr') else 'None'}")
    
    # Load resilience labels
    labels_df = pd.read_csv('node_resilience_labels.csv')
    print(f"  Labels loaded: {len(labels_df)} nodes")
    
    # Add labels to graph data
    data.y = torch.tensor(labels_df['resilient'].values, dtype=torch.long)
    
    # Print class distribution
    resilient_count = (data.y == 1).sum().item()
    vulnerable_count = (data.y == 0).sum().item()
    print(f"\nClass Distribution:")
    print(f"  Resilient (1): {resilient_count} ({resilient_count/len(data.y)*100:.1f}%)")
    print(f"  Vulnerable (0): {vulnerable_count} ({vulnerable_count/len(data.y)*100:.1f}%)")
    
    return data


def create_train_val_test_masks(num_nodes, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Create train/validation/test masks."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Random permutation
    indices = torch.randperm(num_nodes)
    
    # Split indices
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    print(f"\nData Split:")
    print(f"  Training: {train_mask.sum().item()} nodes ({train_ratio*100:.0f}%)")
    print(f"  Validation: {val_mask.sum().item()} nodes ({val_ratio*100:.0f}%)")
    print(f"  Testing: {test_mask.sum().item()} nodes ({test_ratio*100:.0f}%)")
    
    return train_mask, val_mask, test_mask


def train_epoch(model, data, optimizer, train_mask, class_weights=None):
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    out = model(data.x, data.edge_index)
    
    # Calculate loss only on training nodes (with class weights)
    if class_weights is not None:
        loss = F.nll_loss(out[train_mask], data.y[train_mask], weight=class_weights)
    else:
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask):
    """Evaluate model on given mask."""
    model.eval()
    
    # Forward pass
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    
    # Get predictions and labels for masked nodes
    y_true = data.y[mask].cpu().numpy()
    y_pred = pred[mask].cpu().numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return accuracy, precision, recall, f1, y_true, y_pred


def plot_training_history(train_losses, val_accuracies, val_f1_scores):
    """Plot training history."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Training loss
    axes[0].plot(train_losses, label='Training Loss', color='blue')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Over Time (WITH DRNL)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Validation accuracy
    axes[1].plot(val_accuracies, label='Validation Accuracy', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy Over Time (WITH DRNL)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Validation F1 score
    axes[2].plot(val_f1_scores, label='Validation F1 Score', color='orange')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('Validation F1 Score Over Time (WITH DRNL)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history_drnl.png', dpi=300, bbox_inches='tight')
    print("\n✓ Training history plot saved: training_history_drnl.png")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix (WITH DRNL)'):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Vulnerable', 'Resilient'],
                yticklabels=['Vulnerable', 'Resilient'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    plt.savefig('confusion_matrix_drnl.png', dpi=300, bbox_inches='tight')
    print("✓ Confusion matrix saved: confusion_matrix_drnl.png")
    plt.close()


def main():
    """Main training pipeline."""
    print("="*70)
    print("GNN TRAINING WITH DRNL STRUCTURAL LABELS")
    print("="*70)
    
    # Set device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Load data
    data = load_data()
    data = data.to(device)
    
    # Create train/val/test splits
    train_mask, val_mask, test_mask = create_train_val_test_masks(data.num_nodes)
    
    # Initialize model
    print("\n" + "="*70)
    print("INITIALIZING MODEL")
    print("="*70)
    model = GATResiliencePredictor(
        in_channels=data.x.shape[1],  # Now 11 features (10 + DRNL)
        hidden_channels=64,
        num_heads=4,
        dropout=0.3
    ).to(device)
    print(f"Model: {model.__class__.__name__}")
    print(f"  Input features: {data.x.shape[1]} (includes DRNL!)")
    print(f"  Hidden channels: 64")
    print(f"  Attention heads: 4")
    print(f"  Dropout: 0.3")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # Calculate class weights for imbalanced data
    class_counts = torch.bincount(data.y[train_mask])
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()  # Normalize
    class_weights = class_weights.to(device)
    print(f"\nClass weights: {class_weights.tolist()}")
    
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
    val_accuracies = []
    val_f1_scores = []
    
    for epoch in range(1, num_epochs + 1):
        # Train
        loss = train_epoch(model, data, optimizer, train_mask, class_weights)
        train_losses.append(loss)
        
        # Evaluate on validation set
        val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(model, data, val_mask)
        val_accuracies.append(val_acc)
        val_f1_scores.append(val_f1)
        
        # Print progress every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        # Early stopping based on validation F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_gnn_model_drnl.pt')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch} (best epoch: {best_epoch})")
            break
    
    # Load best model if it was saved
    if best_epoch > 0:
        model.load_state_dict(torch.load('best_gnn_model_drnl.pt', map_location=device))
    else:
        # Save current model if no improvement was made
        torch.save(model.state_dict(), 'best_gnn_model_drnl.pt')
    
    # Plot training history
    plot_training_history(train_losses, val_accuracies, val_f1_scores)
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    # Training set
    train_acc, train_prec, train_rec, train_f1, _, _ = evaluate(model, data, train_mask)
    print(f"\nTraining Set:")
    print(f"  Accuracy:  {train_acc:.4f}")
    print(f"  Precision: {train_prec:.4f}")
    print(f"  Recall:    {train_rec:.4f}")
    print(f"  F1 Score:  {train_f1:.4f}")
    
    # Validation set
    val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(model, data, val_mask)
    print(f"\nValidation Set:")
    print(f"  Accuracy:  {val_acc:.4f}")
    print(f"  Precision: {val_prec:.4f}")
    print(f"  Recall:    {val_rec:.4f}")
    print(f"  F1 Score:  {val_f1:.4f}")
    
    # Test set
    test_acc, test_prec, test_rec, test_f1, y_true, y_pred = evaluate(model, data, test_mask)
    print(f"\nTest Set:")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall:    {test_rec:.4f}")
    print(f"  F1 Score:  {test_f1:.4f}")
    
    # Detailed classification report
    print("\n" + "="*70)
    print("DETAILED CLASSIFICATION REPORT (Test Set)")
    print("="*70)
    print(classification_report(y_true, y_pred, 
                                target_names=['Vulnerable', 'Resilient'],
                                digits=4))
    
    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred)
    
    # Save results
    results = {
        'model': 'GNN_DRNL',
        'best_epoch': best_epoch,
        'train_accuracy': train_acc,
        'train_precision': train_prec,
        'train_recall': train_rec,
        'train_f1': train_f1,
        'val_accuracy': val_acc,
        'val_precision': val_prec,
        'val_recall': val_rec,
        'val_f1': val_f1,
        'test_accuracy': test_acc,
        'test_precision': test_prec,
        'test_recall': test_rec,
        'test_f1': test_f1
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv('training_results_drnl.csv', index=False)
    print("\n✓ Results saved: training_results_drnl.csv")
    
    # Compare with baseline
    try:
        baseline_df = pd.read_csv('training_results.csv')
        baseline_f1 = baseline_df['test_f1'].values[0]
        improvement = ((test_f1 - baseline_f1) / baseline_f1) * 100
        
        print("\n" + "="*70)
        print("COMPARISON WITH BASELINE")
        print("="*70)
        print(f"Baseline GNN (no DRNL): F1 = {baseline_f1:.4f}")
        print(f"Enhanced GNN (with DRNL): F1 = {test_f1:.4f}")
        print(f"Improvement: {improvement:+.2f}%")
        
        if test_f1 > baseline_f1:
            print("\n🎉 DRNL labels improved GNN performance!")
        else:
            print("\n⚠️ DRNL did not improve performance (may need tuning)")
    except:
        print("\nNote: Could not load baseline results for comparison")
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - best_gnn_model_drnl.pt (trained model with DRNL)")
    print("  - training_history_drnl.png (loss/accuracy plots)")
    print("  - confusion_matrix_drnl.png (test set confusion matrix)")
    print("  - training_results_drnl.csv (metrics summary)")


if __name__ == "__main__":
    main()
