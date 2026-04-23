"""
Create a 3x2 grid of Validation F1 Score curves from individual training history files.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)
plt.rcParams['font.size'] = 11

# Model configurations
models = [
    ('GAT', 'gat_training_history.csv', '#1f77b4'),
    ('GCN', 'gcn_training_history.csv', '#ff7f0e'),
    ('GraphSAGE', 'graphsage_training_history.csv', '#2ca02c'),
    ('GIN', 'gin_training_history.csv', '#d62728'),
    ('TransformerConv', 'transformerconv_training_history.csv', '#9467bd'),
    ('GINE', 'gine_training_history.csv', '#8c564b')
]

# Create 3x2 subplot grid
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, (model_name, history_file, color) in enumerate(models):
    ax = axes[idx]
    
    # Check if file exists
    if not os.path.exists(history_file):
        ax.text(0.5, 0.5, f'{model_name}\nData not found', 
                ha='center', va='center', fontsize=14, color='red')
        ax.set_title(model_name, fontsize=14, fontweight='bold')
        continue
    
    # Load training history
    df = pd.read_csv(history_file)
    
    # Determine which column contains validation F1
    val_f1_col = None
    for col in df.columns:
        if 'val' in col.lower() and 'f1' in col.lower():
            val_f1_col = col
            break
    
    if val_f1_col is None:
        # Try alternative column names
        for col in ['val_f1', 'Val F1', 'validation_f1', 'Validation F1', 'val_f1_score']:
            if col in df.columns:
                val_f1_col = col
                break
    
    if val_f1_col is None:
        ax.text(0.5, 0.5, f'{model_name}\nF1 column not found', 
                ha='center', va='center', fontsize=14, color='red')
        ax.set_title(model_name, fontsize=14, fontweight='bold')
        print(f"⚠️  {model_name}: Available columns: {df.columns.tolist()}")
        continue
    
    # Get epoch column
    epoch_col = 'epoch' if 'epoch' in df.columns else df.columns[0]
    
    # Plot validation F1
    ax.plot(df[epoch_col], df[val_f1_col], color=color, linewidth=2.5, label='Validation F1')
    
    # Find best epoch (highest F1)
    best_epoch = df[val_f1_col].idxmax()
    best_f1 = df[val_f1_col].max()
    
    # Mark best epoch with vertical line
    ax.axvline(x=df[epoch_col].iloc[best_epoch], color='red', linestyle='--', 
               linewidth=1.5, alpha=0.7, label=f'Best Epoch: {df[epoch_col].iloc[best_epoch]}')
    
    # Mark best F1 point
    ax.scatter(df[epoch_col].iloc[best_epoch], best_f1, 
              color='red', s=100, zorder=5, marker='*')
    
    # Formatting
    ax.set_title(f'{model_name} (Best F1: {best_f1:.4f})', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Validation F1 Score', fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)
    
    print(f"✓ {model_name}: Best F1 = {best_f1:.4f} at epoch {df[epoch_col].iloc[best_epoch]}")

# Overall title
fig.suptitle('Training Convergence Analysis: Validation F1 Scores Across All GNN Architectures', 
             fontsize=16, fontweight='bold', y=0.995)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.99])

# Save figure
output_file = 'all_models_convergence_grid.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✅ Saved convergence grid to: {output_file}")

plt.close()

print("\n" + "="*70)
print("CONVERGENCE GRID CREATION COMPLETE!")
print("="*70)
print(f"\nGenerated file: {output_file}")
print("\nThis figure shows:")
print("  • Validation F1 score progression for all 6 models")
print("  • Best epoch marked with red dashed line and star")
print("  • Convergence speed comparison across architectures")
print("  • Ready for inclusion in your Results section!")
