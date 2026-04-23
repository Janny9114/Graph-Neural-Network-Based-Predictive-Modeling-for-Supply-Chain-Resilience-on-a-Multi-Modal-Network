"""
Create Comprehensive Cross-Validation Comparison Table
Combines GNN and ML baseline results for academic paper

This script:
1. Loads CV results from both GNN and ML models
2. Creates a unified comparison table
3. Generates publication-ready tables and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_cv_results():
    """Load all cross-validation results."""
    results = {}
    
    # GNN Models
    gnn_models = ['GAT', 'GCN', 'GraphSAGE', 'GIN', 'GINE', 'TransformerConv']
    for model in gnn_models:
        filename = f'{model.lower()}_cv_results.csv'
        if os.path.exists(filename):
            results[model] = pd.read_csv(filename)
            print(f"✓ Loaded {model}")
        else:
            print(f"⚠ Missing {filename}")
    
    # ML Models
    ml_models = ['Random Forest', 'XGBoost', 'Gradient Boosting', 'Logistic Regression', 'SVM']
    for model in ml_models:
        filename = f'{model.lower().replace(" ", "_")}_cv_results.csv'
        if os.path.exists(filename):
            results[model] = pd.read_csv(filename)
            print(f"✓ Loaded {model}")
        else:
            print(f"⚠ Missing {filename}")
    
    return results


def create_summary_table(results):
    """Create summary statistics table."""
    summary_data = []
    
    for model_name, df in results.items():
        summary_data.append({
            'Model': model_name,
            'Type': 'GNN' if model_name in ['GAT', 'GCN', 'GraphSAGE', 'GIN', 'GINE', 'TransformerConv'] else 'ML Baseline',
            'Accuracy': f"{df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}",
            'Precision': f"{df['precision'].mean():.4f} ± {df['precision'].std():.4f}",
            'Recall': f"{df['recall'].mean():.4f} ± {df['recall'].std():.4f}",
            'F1 Score': f"{df['f1'].mean():.4f} ± {df['f1'].std():.4f}",
            'F1_mean': df['f1'].mean(),  # For sorting
            'F1_std': df['f1'].std()
        })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary = df_summary.sort_values('F1_mean', ascending=False)
    
    return df_summary


def create_latex_table(df_summary):
    """Create LaTeX-formatted table for academic paper."""
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{5-Fold Cross-Validation Results: GNN vs. ML Baselines}")
    latex_lines.append("\\label{tab:cv_results}")
    latex_lines.append("\\begin{tabular}{llcccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("\\textbf{Model} & \\textbf{Type} & \\textbf{Accuracy} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1 Score} \\\\")
    latex_lines.append("\\midrule")
    
    for _, row in df_summary.iterrows():
        latex_lines.append(f"{row['Model']} & {row['Type']} & {row['Accuracy']} & {row['Precision']} & {row['Recall']} & {row['F1 Score']} \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    return "\n".join(latex_lines)


def create_comparison_plot(results):
    """Create comprehensive comparison visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # Prepare data
        plot_data = []
        for model_name, df in results.items():
            model_type = 'GNN' if model_name in ['GAT', 'GCN', 'GraphSAGE', 'GIN', 'GINE', 'TransformerConv'] else 'ML Baseline'
            for _, row in df.iterrows():
                plot_data.append({
                    'Model': model_name,
                    'Type': model_type,
                    'Value': row[metric]
                })
        
        df_plot = pd.DataFrame(plot_data)
        
        # Sort by mean performance
        model_order = df_plot.groupby('Model')['Value'].mean().sort_values(ascending=False).index.tolist()
        
        # Create box plot with color coding
        colors = ['#2ecc71' if model in ['GAT', 'GCN', 'GraphSAGE', 'GIN', 'GINE', 'TransformerConv'] else '#e74c3c' 
                  for model in model_order]
        
        bp = ax.boxplot([df_plot[df_plot['Model'] == model]['Value'].values for model in model_order],
                        labels=model_order,
                        patch_artist=True,
                        showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
        
        # Color boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_title(f'{metric.capitalize()} Comparison (5-Fold CV)', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#2ecc71', alpha=0.6, label='GNN Models'),
                          Patch(facecolor='#e74c3c', alpha=0.6, label='ML Baselines')]
        ax.legend(handles=legend_elements, loc='lower left')
    
    plt.tight_layout()
    plt.savefig('comprehensive_cv_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: comprehensive_cv_comparison.png")
    plt.close()


def create_ranking_table(df_summary):
    """Create ranking table by F1 score."""
    ranking = df_summary[['Model', 'Type', 'F1_mean', 'F1_std']].copy()
    ranking['Rank'] = range(1, len(ranking) + 1)
    ranking['F1 Score'] = ranking.apply(lambda x: f"{x['F1_mean']:.4f} ± {x['F1_std']:.4f}", axis=1)
    ranking = ranking[['Rank', 'Model', 'Type', 'F1 Score']]
    
    return ranking


def main():
    """Main execution."""
    print("="*70)
    print("COMPREHENSIVE CROSS-VALIDATION COMPARISON")
    print("="*70)
    
    # Load results
    print("\nLoading cross-validation results...")
    results = load_cv_results()
    
    if not results:
        print("\n❌ No CV results found. Please run cross-validation scripts first:")
        print("  - train_gnn_with_cross_validation.py")
        print("  - train_ml_with_cross_validation.py")
        return
    
    print(f"\n✓ Loaded {len(results)} models")
    
    # Create summary table
    print("\nCreating summary table...")
    df_summary = create_summary_table(results)
    
    # Display table
    print("\n" + "="*70)
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print("="*70)
    print("\n" + df_summary[['Model', 'Type', 'Accuracy', 'Precision', 'Recall', 'F1 Score']].to_string(index=False))
    
    # Save CSV
    df_summary.to_csv('comprehensive_cv_summary.csv', index=False)
    print("\n✓ Saved: comprehensive_cv_summary.csv")
    
    # Create ranking table
    print("\n" + "="*70)
    print("MODEL RANKING (by F1 Score)")
    print("="*70)
    ranking = create_ranking_table(df_summary)
    print("\n" + ranking.to_string(index=False))
    ranking.to_csv('model_ranking.csv', index=False)
    print("\n✓ Saved: model_ranking.csv")
    
    # Create LaTeX table
    print("\n" + "="*70)
    print("LATEX TABLE (for academic paper)")
    print("="*70)
    latex_table = create_latex_table(df_summary)
    print("\n" + latex_table)
    
    with open('cv_results_latex_table.tex', 'w') as f:
        f.write(latex_table)
    print("\n✓ Saved: cv_results_latex_table.tex")
    
    # Create comparison plot
    print("\n" + "="*70)
    print("CREATING COMPARISON VISUALIZATION")
    print("="*70)
    create_comparison_plot(results)
    
    # Statistical analysis
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS")
    print("="*70)
    
    gnn_f1 = [df['f1'].mean() for name, df in results.items() 
              if name in ['GAT', 'GCN', 'GraphSAGE', 'GIN', 'GINE', 'TransformerConv']]
    ml_f1 = [df['f1'].mean() for name, df in results.items() 
             if name not in ['GAT', 'GCN', 'GraphSAGE', 'GIN', 'GINE', 'TransformerConv']]
    
    if gnn_f1 and ml_f1:
        print(f"\nGNN Models:")
        print(f"  Mean F1: {np.mean(gnn_f1):.4f}")
        print(f"  Std F1:  {np.std(gnn_f1):.4f}")
        print(f"  Best:    {max(gnn_f1):.4f}")
        
        print(f"\nML Baselines:")
        print(f"  Mean F1: {np.mean(ml_f1):.4f}")
        print(f"  Std F1:  {np.std(ml_f1):.4f}")
        print(f"  Best:    {max(ml_f1):.4f}")
        
        improvement = ((np.mean(gnn_f1) - np.mean(ml_f1)) / np.mean(ml_f1)) * 100
        print(f"\nGNN Improvement over ML: {improvement:.2f}%")
    
    print("\n" + "="*70)
    print("✅ COMPARISON COMPLETE!")
    print("="*70)
    
    print("\n📁 Output Files:")
    print("  - comprehensive_cv_summary.csv")
    print("  - model_ranking.csv")
    print("  - cv_results_latex_table.tex")
    print("  - comprehensive_cv_comparison.png")


if __name__ == "__main__":
    main()
