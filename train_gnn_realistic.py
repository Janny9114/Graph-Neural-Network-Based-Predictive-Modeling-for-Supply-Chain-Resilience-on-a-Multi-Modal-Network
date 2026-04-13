"""
GNN Training for Realistic Supply Chain Resilience Prediction
Trains a Graph Attention Network (GAT) on 10,000 realistic disruption scenarios.

Key Features:
- 7 input features: [capacity, cost_factor, risk_level, reliability, x, y, buffer]
- ALL nodes know buffer and spatial coordinates
- NO production_impact_pct or is_disrupted flags (hidden)
- GNN must infer hidden impact from graph structure
- Expected to significantly outperform ML models
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from tqdm import tqdm

class GATResiliencePredictor(torch.nn.Module):
    """
    Graph Attention Network for predicting supply chain node resilience.
    
    Architecture:
    - 3 GAT layers with multi-head attention
    - Batch normalization for stability
    - Dropout for regularization
    - Binary classification output
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


def load_scenario_data(scenario_dir='scenario_graphs_realistic'):
    """Load all scenario graphs from directory."""
    print("="*70)
    print("LOADING REALISTIC SCENARIO DATA")
    print("="*70)
    
    # Load metadata
    metadata_path = os.path.join(scenario_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"\nMetadata:")
    print(f"  Total scenarios: {metadata['num_scenarios']}")
    print(f"  Nodes per graph: {metadata['num_nodes']}")
    print(f"  Features per node: {metadata['num_features']}")
    print(f"  Feature names: {metadata['feature_names']}")
    
    print(f"\n📋 Feature Description:")
    for name, desc in metadata['feature_description'].items():
        print(f"  {name}: {desc}")
    
    # Load all scenario files
    print(f"\n📂 Loading {metadata['num_scenarios']} scenario graphs...")
    scenarios = []
    
    for i in tqdm(range(metadata['num_scenarios']), desc="Loading scenarios"):
        scenario_path = os.path.join(scenario_dir, f'scenario_{i:05d}.pt')
        data = torch.load(scenario_path, weights_only=False)
        scenarios.append(data)
    
    print(f"  ✓ Loaded {len(scenarios)} scenarios")
    
    # Analyze label distribution
    total_labeled = sum(data.train_mask.sum().item() for data in scenarios)
    total_resilient = sum(((data.y == 1) & data.train_mask).sum().item() for data in scenarios)
    total_vulnerable = sum(((data.y == 0) & data.train_mask).sum().item() for data in scenarios)
    
    print(f"\n📊 Label Distribution:")
    print(f"  Total labeled nodes: {total_labeled:,}")
    print(f"  Resilient (1): {total_resilient:,} ({total_resilient/total_labeled*100:.1f}%)")
    print(f"  Vulnerable (0): {total_vulnerable:,} ({total_vulnerable/total_labeled*100:.1f}%)")
    
    return scenarios, metadata


def split_scenarios(scenarios, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Split scenarios into train/val/test sets."""
    print("\n" + "="*70)
    print("SPLITTING SCENARIOS")
    print("="*70)
    
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
    
    print(f"\n  Train: {len(train_scenarios)} scenarios")
    print(f"  Val:   {len(val_scenarios)} scenarios")
    print(f"  Test:  {len(test_scenarios)} scenarios")
    
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
        
        # Forward pass
        out = model(data.x, data.edge_index)
        
        # Compute loss with class weights (weight vulnerable class 2x)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask], weight=class_weights)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * data.train_mask.sum().item()
        pred = out[data.train_mask].argmax(dim=1)
        total_correct += (pred == data.y[data.train_mask]).sum().item()
        total_samples += data.train_mask.sum().item()
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    all_preds = []
    all_labels = []
    
    for data in loader:
        data = data.to(device)
        
        # Forward pass
        out = model(data.x, data.edge_index)
        
        # Compute loss
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        
        # Track metrics
        total_loss += loss.item() * data.train_mask.sum().item()
        pred = out[data.train_mask].argmax(dim=1)
        total_correct += (pred == data.y[data.train_mask]).sum().item()
        total_samples += data.train_mask.sum().item()
        
        # Store predictions
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(data.y[data.train_mask].cpu().numpy())
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    # Calculate detailed metrics
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1, all_preds, all_labels


def train_model(model, train_loader, val_loader, optimizer, device, class_weights, num_epochs=100, patience=15):
    """Train model with early stopping and class weights."""
    print("\n" + "="*70)
    print("TRAINING GNN MODEL WITH CLASS WEIGHTS")
    print("="*70)
    print(f"  Class weights: Vulnerable={class_weights[0]:.1f}, Resilient={class_weights[1]:.1f}")
    
    best_val_f1 = 0
    patience_counter = 0
    history = []
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, class_weights)
        
        # Validate
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(model, val_loader, device)
        
        # Save history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_precision': val_prec,
            'val_recall': val_rec,
            'val_f1': val_f1
        })
        
        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), 'best_gnn_model_realistic.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠ Early stopping at epoch {epoch} (patience={patience})")
                break
    
    print(f"\n✓ Training complete!")
    print(f"  Best validation F1: {best_val_f1:.4f}")
    
    return history


def plot_training_history(history):
    """Plot training history."""
    df = pd.DataFrame(history)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(df['epoch'], df['train_acc'], label='Train Acc', marker='o')
    axes[0, 1].plot(df['epoch'], df['val_acc'], label='Val Acc', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1, 0].plot(df['epoch'], df['val_f1'], label='Val F1', marker='s', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Validation F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Precision & Recall
    axes[1, 1].plot(df['epoch'], df['val_precision'], label='Precision', marker='o')
    axes[1, 1].plot(df['epoch'], df['val_recall'], label='Recall', marker='s')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Validation Precision & Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history_realistic.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved training history: training_history_realistic.png")
    plt.close()


def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Vulnerable', 'Resilient'],
                yticklabels=['Vulnerable', 'Resilient'])
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_realistic.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved confusion matrix: confusion_matrix_realistic.png")
    plt.close()


def main():
    """Main training pipeline."""
    print("="*70)
    print("GNN TRAINING - REALISTIC SCENARIOS")
    print("="*70)
    print("\nKey Features (HARDER VERSION):")
    print("  ✓ 7 input features: [capacity, cost_factor, risk_level, reliability, x, y, buffer]")
    print("  ✓ ALL nodes know buffer and spatial coordinates (realistic)")
    print("  ✓ NO production_impact_pct (hidden from everyone)")
    print("  ✓ NO is_disrupted flag (must infer from graph)")
    print("  ✓ GNN MUST use graph structure to infer everything")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    # Load data
    # Change this line:
    scenarios, metadata = load_scenario_data('scenario_graphs_edge_disruptions')

    
    # Split data
    train_scenarios, val_scenarios, test_scenarios = split_scenarios(scenarios)
    
    # Create data loaders
    print("\n" + "="*70)
    print("CREATING DATA LOADERS")
    print("="*70)
    
    train_loader = DataLoader(train_scenarios, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_scenarios, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_scenarios, batch_size=32, shuffle=False)
    
    print(f"\n  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    
    # Initialize model
    print("\n" + "="*70)
    print("INITIALIZING MODEL")
    print("="*70)
    
    in_channels = metadata['num_features']
    model = GATResiliencePredictor(
        in_channels=in_channels,
        hidden_channels=64,
        num_heads=4,
        dropout=0.3
    ).to(device)
    
    print(f"\n  Model: GAT with {in_channels} input features")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    
    # Calculate class weights (weight vulnerable class 1.3x to address imbalance)
    class_weights = torch.tensor([1.3, 1.0], dtype=torch.float).to(device)
    print(f"\n  Class weights: Vulnerable=1.3, Resilient=1.0")
    
    # Train model
    history = train_model(model, train_loader, val_loader, optimizer, device, class_weights, num_epochs=100, patience=15)
    
    # Save training history
    df_history = pd.DataFrame(history)
    df_history.to_csv('training_results_realistic.csv', index=False)
    print(f"\n✓ Saved training results: training_results_realistic.csv")
    
    # Plot training history
    plot_training_history(history)
    
    # Load best model
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)
    
    model.load_state_dict(torch.load('best_gnn_model_realistic.pt'))
    
    # Evaluate on test set
    test_loss, test_acc, test_prec, test_rec, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, device
    )
    
    print(f"\n📊 Test Set Results:")
    print(f"  Loss:      {test_loss:.4f}")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall:    {test_rec:.4f}")
    print(f"  F1 Score:  {test_f1:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(test_labels, test_preds)
    
    # Print classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(test_labels, test_preds, 
                                target_names=['Vulnerable', 'Resilient'],
                                digits=4))
    
    # Save GNN results for ML benchmark comparison
    gnn_results = {
        'model': 'GNN (GAT)',
        'accuracy': test_acc,
        'precision': test_prec,
        'recall': test_rec,
        'f1': test_f1
    }
    
    import json
    with open('gnn_results_realistic.json', 'w') as f:
        json.dump(gnn_results, f, indent=2)
    
    print(f"\n✓ Saved GNN results: gnn_results_realistic.json")
    
    # Final summary
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    
    print(f"\n📁 Saved Files:")
    print(f"  ✓ best_gnn_model_realistic.pt - Best model weights")
    print(f"  ✓ training_results_realistic.csv - Training history")
    print(f"  ✓ training_history_realistic.png - Training curves")
    print(f"  ✓ confusion_matrix_realistic.png - Confusion matrix")
    
    print(f"\n🎯 Next Steps:")
    print(f"  1. Run benchmark: python benchmark_ml_realistic.py")
    print(f"  2. Compare GNN vs ML performance")
    print(f"  3. Analyze why GNN outperforms (graph reasoning)")
    
    print(f"\n💡 Expected:")
    print(f"  - GNN should achieve 85-95% accuracy")
    print(f"  - ML models should achieve 65-75% accuracy")
    print(f"  - GNN leverages graph structure to infer hidden production_impact_pct")


if __name__ == "__main__":
    main()
