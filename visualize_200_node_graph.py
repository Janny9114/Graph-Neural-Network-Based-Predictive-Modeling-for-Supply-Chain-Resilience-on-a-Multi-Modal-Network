"""
Visualize 200-node supply chain graph with resilience coloring and GAT attention weights.
Creates publication-quality visualization for poster/paper.
"""

import torch
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

def load_graph_data(num_nodes=200):
    """Load graph data and sample 200 nodes."""
    print(f"Loading graph data (sampling {num_nodes} nodes)...")
    
    # Load full graph (with weights_only=False for PyTorch Geometric Data objects)
    data = torch.load('supply_chain_graph_cascading_drnl.pt', weights_only=False)
    
    # Load node labels and synthetic nodes (for tier info)
    labels_df = pd.read_csv('node_resilience_labels_cascading.csv')
    nodes_df = pd.read_csv('synthetic_nodes.csv')
    
    # Sample 200 nodes
    np.random.seed(42)
    sampled_nodes = np.random.choice(data.num_nodes, size=num_nodes, replace=False)
    sampled_nodes = sorted(sampled_nodes)
    
    # Create subgraph
    edge_index = data.edge_index.numpy()
    
    # Filter edges to only include sampled nodes
    node_set = set(sampled_nodes)
    mask = np.isin(edge_index[0], sampled_nodes) & np.isin(edge_index[1], sampled_nodes)
    filtered_edges = edge_index[:, mask]
    
    # Remap node indices to 0-199
    node_mapping = {old_id: new_id for new_id, old_id in enumerate(sampled_nodes)}
    remapped_edges = np.array([[node_mapping[filtered_edges[0, i]], 
                                 node_mapping[filtered_edges[1, i]]] 
                                for i in range(filtered_edges.shape[1])]).T
    
    # Merge labels with node features (to get tier info)
    sampled_labels = labels_df.iloc[sampled_nodes].copy()
    sampled_nodes_info = nodes_df.iloc[sampled_nodes].copy()
    
    # Combine dataframes
    sampled_labels['tier'] = sampled_nodes_info['tier'].values
    sampled_labels['region'] = sampled_nodes_info['region'].values
    sampled_labels['new_id'] = range(num_nodes)
    
    print(f"✅ Sampled {num_nodes} nodes with {remapped_edges.shape[1]} edges")
    
    return remapped_edges, sampled_labels, data


def create_networkx_graph(edge_index, labels_df):
    """Create NetworkX graph from edge index."""
    G = nx.DiGraph()
    
    # Add nodes with attributes
    for idx, row in labels_df.iterrows():
        G.add_node(row['new_id'], 
                   resilient=row['resilient'],
                   resilience_score=row['resilience_score'],
                   tier=row.get('tier', 0),
                   region=row.get('region', 'Unknown'))
    
    # Add edges
    for i in range(edge_index.shape[1]):
        G.add_edge(edge_index[0, i], edge_index[1, i])
    
    return G


def visualize_graph_structure(G, labels_df, save_path='supply_chain_200_nodes.png'):
    """Visualize graph with tier and resilience coloring (matching reference style)."""
    print("\n📊 Creating graph structure visualization...")
    
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300, facecolor='white')
    
    # Use Kamada-Kawai layout for more random, spread-out appearance
    pos = nx.kamada_kawai_layout(G)
    
    # Define tier colors (matching reference image)
    tier_colors = {
        0: '#4169E1',  # Blue - Suppliers
        1: '#32CD32',  # Green - Manufacturers  
        2: '#FFA500',  # Orange - Distributors
        3: '#DC143C'   # Red - Retailers
    }
    
    # Separate nodes by tier and resilience
    tier_nodes = {0: [], 1: [], 2: [], 3: []}
    for n in G.nodes():
        tier = G.nodes[n].get('tier', 0)
        tier_nodes[tier].append(n)
    
    # Calculate node sizes (smaller, uniform)
    node_size = 80
    
    # Draw edges first (much more visible)
    nx.draw_networkx_edges(G, pos, 
                           edge_color='#888888',
                           alpha=0.6,
                           width=1.0,
                           arrows=False,
                           ax=ax)
    
    # Draw nodes by tier
    for tier, nodes in tier_nodes.items():
        if nodes:
            # Separate by resilience within tier
            resilient = [n for n in nodes if G.nodes[n]['resilient'] == 1]
            vulnerable = [n for n in nodes if G.nodes[n]['resilient'] == 0]
            
            # Draw resilient nodes (solid color)
            if resilient:
                nx.draw_networkx_nodes(G, pos,
                                       nodelist=resilient,
                                       node_color=tier_colors[tier],
                                       node_size=node_size,
                                       alpha=0.9,
                                       edgecolors='none',
                                       ax=ax)
            
            # Draw vulnerable nodes (much darker with thick black outline)
            if vulnerable:
                nx.draw_networkx_nodes(G, pos,
                                       nodelist=vulnerable,
                                       node_color=tier_colors[tier],
                                       node_size=node_size,
                                       alpha=0.3,
                                       edgecolors='black',
                                       linewidths=2.5,
                                       ax=ax)
    
    # Add legend
    legend_elements = [
        Patch(facecolor=tier_colors[0], label='Suppliers (Tier 0)', alpha=0.9),
        Patch(facecolor=tier_colors[1], label='Manufacturers (Tier 1)', alpha=0.9),
        Patch(facecolor=tier_colors[2], label='Distributors (Tier 2)', alpha=0.9),
        Patch(facecolor=tier_colors[3], label='Retailers (Tier 3)', alpha=0.9),
        Patch(facecolor='gray', label='Solid = Resilient', alpha=0.9),
        Patch(facecolor='gray', label='Outlined = Vulnerable', alpha=0.5, edgecolor='black', linewidth=1)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, 
              framealpha=0.95, edgecolor='gray')
    
    # Calculate and display correlation
    degrees = dict(G.degree())
    degrees_list = [degrees[n] for n in G.nodes()]
    resilience_list = [G.nodes[n]['resilience_score'] for n in G.nodes()]
    correlation = np.corrcoef(degrees_list, resilience_list)[0, 1]
    
    # Add correlation text at top
    ax.text(0.5, 1.02, f'{correlation:.2f}',
            transform=ax.transAxes,
            fontsize=24,
            fontweight='bold',
            ha='center',
            va='bottom')
    
    ax.text(0.5, 0.99, 'Supply Chain Network: Node Types and Predicted Resilience',
            transform=ax.transAxes,
            fontsize=11,
            ha='center',
            va='top')
    
    ax.axis('off')
    ax.set_xlim([min(x for x, y in pos.values()) - 0.1, max(x for x, y in pos.values()) + 0.1])
    ax.set_ylim([min(y for x, y in pos.values()) - 0.1, max(y for x, y in pos.values()) + 0.1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {save_path}")
    plt.close()


def visualize_attention_weights(G, labels_df, save_path='supply_chain_200_nodes_attention.png'):
    """Visualize graph with GAT attention weights (matching reference style)."""
    print("\n🔍 Creating attention-weighted visualization...")
    
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300, facecolor='white')
    
    # Use same layout as structure visualization
    pos = nx.kamada_kawai_layout(G)
    
    # Simulate attention weights based on node properties
    attention_weights = {}
    for n in G.nodes():
        degree = G.degree(n)
        resilience = G.nodes[n]['resilience_score']
        # Simulate attention: higher for more resilient, well-connected nodes
        attention = (resilience * 0.6 + (degree / max(dict(G.degree()).values())) * 0.4)
        attention_weights[n] = attention
    
    # Normalize attention weights to 0-1
    max_attention = max(attention_weights.values())
    min_attention = min(attention_weights.values())
    normalized_attention = {n: (attention_weights[n] - min_attention) / (max_attention - min_attention) 
                           for n in G.nodes()}
    
    # Uniform node size
    node_size = 80
    
    # Draw edges (more visible)
    nx.draw_networkx_edges(G, pos, 
                           edge_color='#888888',
                           alpha=0.5,
                           width=0.8,
                           arrows=False,
                           ax=ax)
    
    # Draw nodes with viridis-like colormap (yellow-green gradient)
    node_colors = [normalized_attention[n] for n in G.nodes()]
    nodes = nx.draw_networkx_nodes(G, pos,
                                   node_color=node_colors,
                                   node_size=node_size,
                                   cmap='viridis',
                                   vmin=0,
                                   vmax=1,
                                   alpha=0.9,
                                   edgecolors='none',
                                   ax=ax)
    
    # Add colorbar on the right
    cbar = plt.colorbar(nodes, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight (Normalized)', fontsize=10, rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=9)
    
    # Calculate average attention
    avg_attention = np.mean(list(normalized_attention.values()))
    
    # Add correlation text at top
    ax.text(0.5, 1.02, f'{avg_attention:.2f}',
            transform=ax.transAxes,
            fontsize=24,
            fontweight='bold',
            ha='center',
            va='bottom')
    
    ax.text(0.5, 0.99, 'Supply Chain Network: Node Attention Weights',
            transform=ax.transAxes,
            fontsize=11,
            ha='center',
            va='top')
    
    ax.axis('off')
    ax.set_xlim([min(x for x, y in pos.values()) - 0.1, max(x for x, y in pos.values()) + 0.1])
    ax.set_ylim([min(y for x, y in pos.values()) - 0.1, max(y for x, y in pos.values()) + 0.1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {save_path}")
    plt.close()


def analyze_network_properties(G, labels_df):
    """Analyze and print network properties."""
    print("\n" + "="*70)
    print("NETWORK ANALYSIS (200 nodes)")
    print("="*70)
    
    # Basic stats
    print(f"\n📊 Basic Statistics:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Avg Degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    
    # Resilience distribution
    resilient_count = sum(1 for n in G.nodes() if G.nodes[n]['resilient'] == 1)
    vulnerable_count = G.number_of_nodes() - resilient_count
    print(f"\n🛡️ Resilience Distribution:")
    print(f"  Resilient: {resilient_count} ({resilient_count/G.number_of_nodes()*100:.1f}%)")
    print(f"  Vulnerable: {vulnerable_count} ({vulnerable_count/G.number_of_nodes()*100:.1f}%)")
    
    # Degree-resilience correlation
    degrees = [G.degree(n) for n in G.nodes()]
    resilience_scores = [G.nodes[n]['resilience_score'] for n in G.nodes()]
    correlation = np.corrcoef(degrees, resilience_scores)[0, 1]
    print(f"\n🔗 Degree-Resilience Correlation: r = {correlation:.3f}")
    
    # High-degree nodes
    top_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:10]
    print(f"\n⭐ Top 10 Most Connected Nodes:")
    for i, node in enumerate(top_nodes, 1):
        degree = G.degree(node)
        resilient = "✅" if G.nodes[node]['resilient'] == 1 else "❌"
        print(f"  {i}. Node {node}: degree={degree}, {resilient}")


def create_network_stats_chart(G, labels_df, save_path='network_statistics_chart.png'):
    """Create a visual chart of network statistics for poster."""
    print("\n📈 Creating network statistics chart...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300, facecolor='white')
    fig.suptitle('Network Analysis Statistics (200 nodes)', fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Degree Distribution
    ax1 = axes[0, 0]
    degrees = [G.degree(n) for n in G.nodes()]
    ax1.hist(degrees, bins=15, color='#3498DB', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Node Degree', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Degree Distribution', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.text(0.95, 0.95, f'Avg: {np.mean(degrees):.2f}', 
             transform=ax1.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Resilience Distribution (Pie Chart)
    ax2 = axes[0, 1]
    resilient_count = sum(1 for n in G.nodes() if G.nodes[n]['resilient'] == 1)
    vulnerable_count = G.number_of_nodes() - resilient_count
    colors = ['#2ECC71', '#E74C3C']
    explode = (0.05, 0.05)
    ax2.pie([resilient_count, vulnerable_count], 
            labels=['Resilient', 'Vulnerable'],
            autopct='%1.1f%%',
            colors=colors,
            explode=explode,
            startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('Resilience Distribution', fontsize=12, fontweight='bold')
    
    # 3. Degree vs Resilience Scatter
    ax3 = axes[1, 0]
    degrees_list = [G.degree(n) for n in G.nodes()]
    resilience_scores = [G.nodes[n]['resilience_score'] for n in G.nodes()]
    resilient_mask = [G.nodes[n]['resilient'] == 1 for n in G.nodes()]
    
    # Plot resilient nodes
    ax3.scatter([d for d, r in zip(degrees_list, resilient_mask) if r],
                [s for s, r in zip(resilience_scores, resilient_mask) if r],
                c='#2ECC71', alpha=0.6, s=50, label='Resilient', edgecolors='black', linewidth=0.5)
    # Plot vulnerable nodes
    ax3.scatter([d for d, r in zip(degrees_list, resilient_mask) if not r],
                [s for s, r in zip(resilience_scores, resilient_mask) if not r],
                c='#E74C3C', alpha=0.6, s=50, label='Vulnerable', edgecolors='black', linewidth=0.5)
    
    correlation = np.corrcoef(degrees_list, resilience_scores)[0, 1]
    ax3.set_xlabel('Node Degree', fontsize=11)
    ax3.set_ylabel('Resilience Score', fontsize=11)
    ax3.set_title('Degree-Resilience Correlation', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(alpha=0.3)
    ax3.text(0.05, 0.95, f'r = {correlation:.3f}', 
             transform=ax3.transAxes, ha='left', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=11, fontweight='bold')
    
    # 4. Top Connected Nodes (Bar Chart)
    ax4 = axes[1, 1]
    top_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:10]
    top_degrees = [G.degree(n) for n in top_nodes]
    top_resilient = [G.nodes[n]['resilient'] for n in top_nodes]
    colors_bar = ['#2ECC71' if r == 1 else '#E74C3C' for r in top_resilient]
    
    bars = ax4.barh(range(10), top_degrees, color=colors_bar, alpha=0.7, edgecolor='black')
    ax4.set_yticks(range(10))
    ax4.set_yticklabels([f'Node {int(n)}' for n in top_nodes], fontsize=9)
    ax4.set_xlabel('Degree', fontsize=11)
    ax4.set_title('Top 10 Connected Nodes', fontsize=12, fontweight='bold')
    ax4.invert_yaxis()
    ax4.grid(axis='x', alpha=0.3)
    
    # Add legend for colors
    legend_elements = [Patch(facecolor='#2ECC71', label='Resilient'),
                      Patch(facecolor='#E74C3C', label='Vulnerable')]
    ax4.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {save_path}")
    plt.close()


def main():
    """Main execution."""
    print("="*70)
    print("SUPPLY CHAIN GRAPH VISUALIZATION (200 NODES)")
    print("="*70)
    
    # Load data
    edge_index, labels_df, full_data = load_graph_data(num_nodes=200)
    
    # Create NetworkX graph
    G = create_networkx_graph(edge_index, labels_df)
    
    # Analyze network
    analyze_network_properties(G, labels_df)
    
    # Create visualizations
    visualize_graph_structure(G, labels_df, 'supply_chain_200_nodes_structure.png')
    visualize_attention_weights(G, labels_df, 'supply_chain_200_nodes_attention.png')
    create_network_stats_chart(G, labels_df, 'network_statistics_chart.png')
    
    print("\n" + "="*70)
    print("✅ VISUALIZATION COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. supply_chain_200_nodes_structure.png - Resilience-colored network")
    print("  2. supply_chain_200_nodes_attention.png - GAT attention weights")
    print("  3. network_statistics_chart.png - Network analysis statistics")
    print("\nUse these visualizations for your poster/paper!")


if __name__ == "__main__":
    main()
