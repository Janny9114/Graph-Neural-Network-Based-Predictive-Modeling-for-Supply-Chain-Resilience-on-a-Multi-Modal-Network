"""
GNN Training for Supply Chain Resilience Prediction - Scenario-Based Learning
Trains a Graph Attention Network (GAT) on 1000 separate disruption scenarios.
Fixes aggregation loss by learning from scenario-specific labels.
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


def load_scenario_data(scenario_dir='scenario_graphs_tts_ttr'):
    """Load all scenario graphs from directory."""
    print("="*70)
    print("LOADING TTS vs TTR SCENARIO DATA")
    print("="*70)
    
    # Load metadata
    metadata_path = os.path.join(scenario_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"\nMetadata:")
    print(f"  Total scenarios: {metadata['num_scenarios']}")
    print(f"  Nodes per graph: {metadata['num_nodes']}")
    print(f"  Edges per graph: {metadata['num_edges']}")
    print(f"  Features per node: {metadata['num_features']}")
    print(f"  Feature names: {metadata['feature_names']}")
    
    # Load all scenario files
    print(f"\n📂 Loading {metadata['num_scenarios']} scenario graphs...")
    scenarios = []
    
    for i in tqdm(range(metadata['num_scenarios']), desc="Loading scenarios"):
        scenario_path = os.path.join(scenario_dir, f'scenario_{i:05d}.pt')
        data = torch.load(scenario_path, weights_only=False)
        scenarios.append(data)
    
    print(f"✓ Loaded {len(scenarios)} scenarios")
    
    # Calculate statistics
    total_labeled = sum((s.train_mask.sum().item() for s in scenarios))
    total_resilient = sum(((s.y[s.train_mask] == 1).sum().item() for s in scenarios))
    total_vulnerable = sum(((s.y[s.train_mask] == 0).sum().item() for s in scenarios))
    
    print(f"\nDataset Statistics:")
    print(f"  Total labeled nodes (across all scenarios): {total_labeled:,}")
    print(f"  Resilient (1): {total_resilient:,} ({total_resilient/total_labeled*100:.1f}%)")
    print(f"  Vulnerable (0): {total_vulnerable:,} ({total_vulnerable/total_labeled*100:.1f}%)")
    
    return scenarios, metadata


def split_scenarios(scenarios, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Split scenarios into train/val/test sets."""
    torch.manual_seed(seed)
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
    
    print(f"\nScenario Split:")
    print(f"  Train: {len(train_scenarios)} scenarios ({len(train_scenarios)/num_scenarios*100:.1f}%)")
    print(f"  Val: {len(val_scenarios)} scenarios ({len(val_scenarios)/num_scenarios*100:.1f}%)")
    print(f"  Test: {len(test_scenarios)} scenarios ({len(test_scenarios)/num_scenarios*100:.1f}%)")
    
    return train_scenarios, val_scenarios, test_scenarios


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data.x, data.edge_index)
        
        # Only compute loss on labeled nodes (train_mask)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        
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
        
        # Compute loss on labeled nodes
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        
        # Track metrics
        total_loss += loss.item() * data.train_mask.sum().item()
        pred = out[data.train_mask].argmax(dim=1)
        total_correct += (pred == data.y[data.train_mask]).sum().item()
        total_samples += data.train_mask.sum().item()
        
        # Collect predictions and labels
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(data.y[data.train_mask].cpu().numpy())
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    # Calculate additional metrics
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1, all_preds, all_labels


def train_model(model, train_loader, val_loader, optimizer, device, num_epochs=100, patience=15):
    """Train model with early stopping."""
    print("\n" + "="*70)
    print("TRAINING MODEL")
    print("="*70)
    
    best_val_loss = float('inf')
    best_val_acc = 0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': []
    }
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(model, val_loader, device)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_prec)
        history['val_recall'].append(val_rec)
        history['val_f1'].append(val_f1)
        
        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Val F1: {val_f1:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_gnn_model_scenarios.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠ Early stopping triggered at epoch {epoch}")
                break
    
    print(f"\n✓ Training complete!")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Best validation accuracy: {best_val_acc:.4f}")
    
    return history


def plot_training_history(history):
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[0, 1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision, Recall, F1
    axes[1, 0].plot(history['val_precision'], label='Precision', linewidth=2)
    axes[1, 0].plot(history['val_recall'], label='Recall', linewidth=2)
    axes[1, 0].plot(history['val_f1'], label='F1 Score', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Validation Metrics')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Final metrics summary
    axes[1, 1].axis('off')
    final_metrics = f"""
    Final Validation Metrics:
    
    Accuracy:  {history['val_acc'][-1]:.4f}
    Precision: {history['val_precision'][-1]:.4f}
    Recall:    {history['val_recall'][-1]:.4f}
    F1 Score:  {history['val_f1'][-1]:.4f}
    
    Training Epochs: {len(history['train_loss'])}
    """
    axes[1, 1].text(0.1, 0.5, final_metrics, fontsize=14, family='monospace',
                    verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('training_history_scenarios.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved training history plot: training_history_scenarios.png")
    plt.close()


def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Vulnerable', 'Resilient'],
                yticklabels=['Vulnerable', 'Resilient'])
    plt.title('Confusion Matrix - Scenario-Based Training')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_scenarios.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved confusion matrix: confusion_matrix_scenarios.png")
    plt.close()


def main():
    """Main training pipeline."""
    print("="*70)
    print("GNN TRAINING - SCENARIO-BASED LEARNING")
    print("="*70)
    print("\nThis script trains on 1000 separate disruption scenarios")
    print("to learn conditional resilience patterns.\n")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load scenarios
    scenarios, metadata = load_scenario_data('scenario_graphs_tts_ttr')
    
    # Split scenarios
    train_scenarios, val_scenarios, test_scenarios = split_scenarios(
        scenarios, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42
    )
    
    # Create data loaders
    print("\n📦 Creating data loaders...")
    train_loader = DataLoader(train_scenarios, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_scenarios, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_scenarios, batch_size=32, shuffle=False)
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Initialize model
    print("\n🧠 Initializing model...")
    in_channels = metadata['num_features']
    model = GATResiliencePredictor(
        in_channels=in_channels,
        hidden_channels=64,
        num_heads=4,
        dropout=0.3
    ).to(device)
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    
    # Train model
    history = train_model(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=100, patience=15
    )
    
    # Load best model
    print("\n📊 Evaluating best model on test set...")
    model.load_state_dict(torch.load('best_gnn_model_scenarios.pt'))
    
    # Test evaluation
    test_loss, test_acc, test_prec, test_rec, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, device
    )
    
    print("\n" + "="*70)
    print("TEST SET RESULTS")
    print("="*70)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall: {test_rec:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, 
                                target_names=['Vulnerable', 'Resilient']))
    
    # Plot results
    plot_training_history(history)
    plot_confusion_matrix(test_labels, test_preds)
    
    # Save results
    results_df = pd.DataFrame({
        'epoch': range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'train_acc': history['train_acc'],
        'val_loss': history['val_loss'],
        'val_acc': history['val_acc'],
        'val_precision': history['val_precision'],
        'val_recall': history['val_recall'],
        'val_f1': history['val_f1']
    })
    results_df.to_csv('training_results_scenarios.csv', index=False)
    print(f"\n✓ Saved training results: training_results_scenarios.csv")
    
    print("\n" + "="*70)
    print("✅ SCENARIO-BASED TRAINING COMPLETE!")
    print("="*70)
    print("\nKey Improvements:")
    print("  ✓ Trained on 1000 separate scenarios (no aggregation loss)")
    print("  ✓ Model learns conditional resilience patterns")
    print("  ✓ Each scenario has unique disruption context")
    print("  ✓ GNN discovers network topology effects naturally")


if __name__ == "__main__":
    main()
