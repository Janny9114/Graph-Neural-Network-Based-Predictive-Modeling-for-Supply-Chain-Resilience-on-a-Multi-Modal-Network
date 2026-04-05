import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from typing import Dict, Any, Optional
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from graph_preprocessing import preprocess_supply_chain_data


def load_synthetic_data(
    node_path: str = "synthetic_nodes.csv",
    edge_path: str = "synthetic_edges.csv"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the synthetic supply chain data from CSV files.
    
    Args:
        node_path: Path to the nodes CSV file
        edge_path: Path to the edges CSV file
        
    Returns:
        Tuple of (node_df, edge_df)
    """
    node_df = pd.read_csv(node_path)
    edge_df = pd.read_csv(edge_path)
    
    print(f"Loaded {len(node_df)} nodes and {len(edge_df)} edges")
    return node_df, edge_df


def prepare_node_features(node_df: pd.DataFrame) -> torch.Tensor:
    """
    Convert node DataFrame to PyTorch tensor for GNN input.
    
    Features included:
    - tier (one-hot encoded, 4 dimensions, NOT standardized)
    - capacity, cost_factor, risk_level, reliability, x, y (Z-score standardized)
    
    Args:
        node_df: DataFrame containing node information
        
    Returns:
        Tensor of shape [num_nodes, num_features]
    """
    # One-hot encode tier (4 dimensions for tiers 0-3)
    tier_values = node_df['tier'].values
    num_nodes = len(node_df)
    num_tiers = 4
    
    # Create one-hot encoding matrix
    tier_one_hot = np.zeros((num_nodes, num_tiers))
    for i, tier in enumerate(tier_values):
        tier_one_hot[i, int(tier)] = 1
    
    # Extract other numerical features
    feature_columns = ['capacity', 'cost_factor', 'risk_level', 'reliability', 'x', 'y']
    numerical_features = node_df[feature_columns].values
    
    # Apply Z-score standardization to all numerical features
    standardized_features = []
    for i, col in enumerate(feature_columns):
        values = numerical_features[:, i]
        mean = values.mean()
        std = values.std()
        
        if std > 0:
            standardized = (values - mean) / std
        else:
            standardized = np.zeros_like(values)
        
        standardized_features.append(standardized)
    
    # Stack standardized features
    standardized_numerical = np.column_stack(standardized_features)
    
    # Concatenate one-hot encoded tier with standardized numerical features
    # Result: [tier_0, tier_1, tier_2, tier_3, capacity_std, cost_factor_std, risk_level_std, reliability_std, x_std, y_std]
    all_features = np.concatenate([tier_one_hot, standardized_numerical], axis=1)
    
    # Convert to tensor
    node_features = torch.tensor(all_features, dtype=torch.float)
    
    print(f"Node features shape: {node_features.shape}")
    print(f"  - Tier (one-hot, NOT standardized): 4 dimensions")
    print(f"  - Numerical features (Z-score standardized): 6 dimensions")
    print(f"    Features: {feature_columns}")
    print(f"    Mean ≈ 0, Std ≈ 1 for each feature")
    print(f"  - Total: {node_features.shape[1]} dimensions")
    
    # Print standardization verification
    print(f"\n  Standardization verification:")
    for i, col in enumerate(feature_columns):
        feat_values = standardized_numerical[:, i]
        print(f"    {col:15s}: mean={feat_values.mean():.6f}, std={feat_values.std():.6f}")
    
    return node_features


def prepare_edge_index(edge_df: pd.DataFrame) -> torch.Tensor:
    """
    Convert edge DataFrame to PyTorch Geometric edge_index format.
    
    Args:
        edge_df: DataFrame containing edge information with 'source' and 'target' columns
        
    Returns:
        Tensor of shape [2, num_edges] representing edge connectivity
    """
    # Extract source and target nodes
    edge_index = torch.tensor(
        [edge_df['source'].values, edge_df['target'].values],
        dtype=torch.long
    )
    
    print(f"Edge index shape: {edge_index.shape}")
    return edge_index


def prepare_edge_features(edge_df: pd.DataFrame) -> torch.Tensor:
    """
    Convert edge DataFrame to PyTorch tensor for edge attributes.
    
    Features included:
    - lead_time (normalized)
    - transport_cost (normalized)
    - capacity_share
    - disruption_probability
    
    Args:
        edge_df: DataFrame containing edge information
        
    Returns:
        Tensor of shape [num_edges, num_edge_features]
    """
    # Select edge features
    feature_columns = ['lead_time', 'transport_cost', 'capacity_share', 'disruption_probability']
    
    # Extract features
    features = edge_df[feature_columns].values
    
    # Normalize lead_time (min-max scaling)
    lead_time_col = features[:, 0]
    lt_min = lead_time_col.min()
    lt_max = lead_time_col.max()
    features[:, 0] = (lead_time_col - lt_min) / (lt_max - lt_min)
    
    # Normalize transport_cost (min-max scaling)
    cost_col = features[:, 1]
    cost_min = cost_col.min()
    cost_max = cost_col.max()
    features[:, 1] = (cost_col - cost_min) / (cost_max - cost_min)
    
    # Convert to tensor
    edge_features = torch.tensor(features, dtype=torch.float)
    
    print(f"Edge features shape: {edge_features.shape}")
    return edge_features


def encode_regions(node_df: pd.DataFrame) -> tuple[torch.Tensor, Dict[str, int]]:
    """
    Encode region names as integers for categorical features.
    
    Args:
        node_df: DataFrame containing node information with 'region' column
        
    Returns:
        Tuple of (region_tensor, region_mapping)
    """
    # Get unique regions
    unique_regions = node_df['region'].unique()
    region_to_idx = {region: idx for idx, region in enumerate(unique_regions)}
    
    # Encode regions
    region_indices = node_df['region'].map(region_to_idx).values
    region_tensor = torch.tensor(region_indices, dtype=torch.long)
    
    print(f"Number of unique regions: {len(unique_regions)}")
    return region_tensor, region_to_idx


def create_gnn_graph(
    node_path: str = "synthetic_nodes.csv",
    edge_path: str = "synthetic_edges.csv",
    use_preprocessing: bool = True,
    include_edge_features: bool = True
) -> Data:
    """
    Create a PyTorch Geometric Data object from synthetic supply chain data.
    
    Args:
        node_path: Path to the nodes CSV file
        edge_path: Path to the edges CSV file
        use_preprocessing: Whether to use advanced preprocessing (Z-score, log transform)
        include_edge_features: Whether to include edge attributes
        
    Returns:
        PyTorch Geometric Data object ready for GNN training
    """
    # Load data
    node_df, edge_df = load_synthetic_data(node_path, edge_path)
    
    if use_preprocessing:
        # Use advanced preprocessing with Z-score standardization and flow constraints
        print("\n" + "="*70)
        print("Using Advanced Preprocessing (Z-score + Flow Constraints)")
        print("="*70)
        
        processed_node_df, standardized_node_features, standardized_edge_flows = preprocess_supply_chain_data(
            node_df, edge_df,
            adjust_capacity=True,
            verify_constraints=True,
            verbose=True
        )
        
        # Convert to PyTorch tensors
        x = torch.tensor(standardized_node_features, dtype=torch.float32)
        edge_index = prepare_edge_index(edge_df)
        edge_attr = torch.tensor(standardized_edge_flows, dtype=torch.float32).unsqueeze(1)
        
        # Encode regions
        region_labels, region_mapping = encode_regions(node_df)
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            region=region_labels,
            num_nodes=len(node_df)
        )
        
        # Store metadata
        data.region_mapping = region_mapping
        data.node_df = node_df
        data.processed_node_df = processed_node_df
        data.edge_df = edge_df
        data.preprocessing_used = True
        
    else:
        # Use basic preprocessing (original method)
        print("\n" + "="*70)
        print("Using Basic Preprocessing (Min-Max Normalization)")
        print("="*70)
        
        # Prepare node features
        x = prepare_node_features(node_df)
        
        # Prepare edge index
        edge_index = prepare_edge_index(edge_df)
        
        # Prepare edge features (optional)
        edge_attr = None
        if include_edge_features:
            edge_attr = prepare_edge_features(edge_df)
        
        # Encode regions
        region_labels, region_mapping = encode_regions(node_df)
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            region=region_labels,
            num_nodes=len(node_df)
        )
        
        # Store metadata
        data.region_mapping = region_mapping
        data.node_df = node_df
        data.edge_df = edge_df
        data.preprocessing_used = False
    
    print("\n" + "="*70)
    print("GNN GRAPH CONSTRUCTION COMPLETE!")
    print("="*70)
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.edge_index.shape[1]}")
    print(f"Node feature dimension: {data.x.shape[1]}")
    if data.edge_attr is not None:
        print(f"Edge feature dimension: {data.edge_attr.shape[1]}")
    print(f"Number of regions: {len(region_mapping)}")
    print(f"Preprocessing method: {'Advanced (Z-score + Flow)' if use_preprocessing else 'Basic (Min-Max)'}")
    print("="*70)
    
    return data


def visualize_graph_stats(data: Data):
    """
    Print detailed statistics about the constructed graph.
    
    Args:
        data: PyTorch Geometric Data object
    """
    print("\n" + "="*50)
    print("Graph Statistics")
    print("="*50)
    
    # Basic stats
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.edge_index.shape[1]}")
    print(f"Average degree: {data.edge_index.shape[1] / data.num_nodes:.2f}")
    
    # Node feature stats
    print(f"\nNode Features (shape: {data.x.shape}):")
    print(f"  - Mean: {data.x.mean(dim=0).tolist()}")
    print(f"  - Std: {data.x.std(dim=0).tolist()}")
    
    # Edge feature stats
    if data.edge_attr is not None:
        print(f"\nEdge Features (shape: {data.edge_attr.shape}):")
        print(f"  - Mean: {data.edge_attr.mean(dim=0).tolist()}")
        print(f"  - Std: {data.edge_attr.std(dim=0).tolist()}")
    
    # Region distribution
    print(f"\nRegion Distribution:")
    unique_regions, counts = torch.unique(data.region, return_counts=True)
    for region_idx, count in zip(unique_regions.tolist(), counts.tolist()):
        region_name = [k for k, v in data.region_mapping.items() if v == region_idx][0]
        print(f"  - {region_name}: {count} nodes ({count/data.num_nodes*100:.1f}%)")
    
    print("="*50)


def save_graph(data: Data, path: str = "supply_chain_graph.pt"):
    """
    Save the PyTorch Geometric Data object to disk.
    
    Args:
        data: PyTorch Geometric Data object
        path: Path to save the graph
    """
    torch.save(data, path)
    print(f"\nGraph saved to {path}")


def load_graph(path: str = "supply_chain_graph.pt") -> Data:
    """
    Load a saved PyTorch Geometric Data object from disk.
    
    Args:
        path: Path to the saved graph
        
    Returns:
        PyTorch Geometric Data object
    """
    data = torch.load(path)
    print(f"Graph loaded from {path}")
    return data


def visualize_graph(
    data: Data,
    save_path: str = "supply_chain_graph_visualization.png",
    figsize: tuple = (16, 12),
    node_size: int = 300,
    show_labels: bool = False
):
    """
    Visualize the supply chain graph with color-coded tiers and regions.
    
    Args:
        data: PyTorch Geometric Data object
        save_path: Path to save the visualization
        figsize: Figure size (width, height)
        node_size: Size of nodes in the visualization
        show_labels: Whether to show node labels
    """
    print("\nGenerating graph visualization...")
    
    # Convert PyTorch Geometric data to NetworkX
    G = to_networkx(data, to_undirected=False)
    
    # Get node attributes from the stored DataFrames
    node_df = data.node_df
    
    # Create color maps for tiers and regions
    tier_colors = {0: '#FF6B6B', 1: '#4ECDC4', 2: '#45B7D1', 3: '#FFA07A'}
    region_colors = {
        0: '#E74C3C', 1: '#3498DB', 2: '#2ECC71', 
        3: '#F39C12', 4: '#9B59B6', 5: '#1ABC9C'
    }
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Use hierarchical layout based on tiers
    pos = {}
    tier_counts = {}
    
    for node_id, row in node_df.iterrows():
        tier = int(row['tier'])
        if tier not in tier_counts:
            tier_counts[tier] = 0
        
        # Position nodes: x based on tier, y spread within tier
        tier_counts[tier] += 1
        x = tier * 3  # Horizontal spacing between tiers
        
        # Calculate vertical position to spread nodes within tier
        tier_size = len(node_df[node_df['tier'] == tier])
        y = (tier_counts[tier] - tier_size / 2) * 0.5
        
        pos[node_id] = (x, y)
    
    # Visualization 1: Color by Tier
    ax1 = axes[0]
    node_colors_tier = [tier_colors.get(int(node_df.loc[node, 'tier']), '#CCCCCC') 
                        for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors_tier, 
                          node_size=node_size, alpha=0.8, ax=ax1)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3, 
                          arrows=True, arrowsize=10, ax=ax1, 
                          connectionstyle='arc3,rad=0.1')
    
    if show_labels:
        nx.draw_networkx_labels(G, pos, font_size=6, ax=ax1)
    
    ax1.set_title('Supply Chain Graph - Colored by Tier', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Create tier legend
    tier_legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                       markerfacecolor=tier_colors[i], 
                                       markersize=10, label=f'Tier {i}')
                           for i in sorted(tier_colors.keys())]
    ax1.legend(handles=tier_legend_elements, loc='upper left', fontsize=10)
    
    # Visualization 2: Color by Region
    ax2 = axes[1]
    node_colors_region = [region_colors.get(int(data.region[node].item()), '#CCCCCC') 
                         for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors_region, 
                          node_size=node_size, alpha=0.8, ax=ax2)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3, 
                          arrows=True, arrowsize=10, ax=ax2,
                          connectionstyle='arc3,rad=0.1')
    
    if show_labels:
        nx.draw_networkx_labels(G, pos, font_size=6, ax=ax2)
    
    ax2.set_title('Supply Chain Graph - Colored by Region', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Create region legend
    region_legend_elements = []
    for region_name, region_idx in sorted(data.region_mapping.items(), key=lambda x: x[1]):
        region_legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=region_colors.get(region_idx, '#CCCCCC'), 
                      markersize=10, label=region_name)
        )
    ax2.legend(handles=region_legend_elements, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Graph visualization saved to {save_path}")
    plt.close()


def visualize_graph_metrics(
    data: Data,
    save_path: str = "supply_chain_metrics.png",
    figsize: tuple = (16, 10)
):
    """
    Visualize various metrics and distributions of the supply chain graph.
    
    Args:
        data: PyTorch Geometric Data object
        save_path: Path to save the visualization
        figsize: Figure size (width, height)
    """
    print("\nGenerating graph metrics visualization...")
    
    node_df = data.node_df
    edge_df = data.edge_df
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # 1. Node Capacity Distribution by Tier
    ax1 = axes[0, 0]
    for tier in sorted(node_df['tier'].unique()):
        tier_data = node_df[node_df['tier'] == tier]['capacity']
        ax1.hist(tier_data, alpha=0.6, label=f'Tier {tier}', bins=15)
    ax1.set_xlabel('Capacity', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.set_title('Node Capacity Distribution by Tier', fontsize=11, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Risk Level Distribution by Tier
    ax2 = axes[0, 1]
    for tier in sorted(node_df['tier'].unique()):
        tier_data = node_df[node_df['tier'] == tier]['risk_level']
        ax2.hist(tier_data, alpha=0.6, label=f'Tier {tier}', bins=15)
    ax2.set_xlabel('Risk Level', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title('Risk Level Distribution by Tier', fontsize=11, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Region Distribution
    ax3 = axes[0, 2]
    region_counts = node_df['region'].value_counts()
    colors_palette = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']
    ax3.bar(range(len(region_counts)), region_counts.values, 
            color=colors_palette[:len(region_counts)])
    ax3.set_xticks(range(len(region_counts)))
    ax3.set_xticklabels(region_counts.index, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('Number of Nodes', fontsize=10)
    ax3.set_title('Node Distribution by Region', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Lead Time Distribution
    ax4 = axes[1, 0]
    ax4.hist(edge_df['lead_time'], bins=20, color='#4ECDC4', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Lead Time', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.set_title('Edge Lead Time Distribution', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Transport Cost Distribution
    ax5 = axes[1, 1]
    ax5.hist(edge_df['transport_cost'], bins=20, color='#FF6B6B', alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Transport Cost', fontsize=10)
    ax5.set_ylabel('Frequency', fontsize=10)
    ax5.set_title('Edge Transport Cost Distribution', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Disruption Probability Distribution
    ax6 = axes[1, 2]
    ax6.hist(edge_df['disruption_probability'], bins=20, color='#FFA07A', alpha=0.7, edgecolor='black')
    ax6.set_xlabel('Disruption Probability', fontsize=10)
    ax6.set_ylabel('Frequency', fontsize=10)
    ax6.set_title('Edge Disruption Probability Distribution', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Metrics visualization saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    # Create GNN graph from synthetic data
    print("Creating GNN graph from synthetic supply chain data...\n")
    
    # Test with basic preprocessing (one-hot encoding)
    data = create_gnn_graph(
        node_path="synthetic_nodes.csv",
        edge_path="synthetic_edges.csv",
        use_preprocessing=False,
        include_edge_features=True
    )
    
    # Visualize statistics
    visualize_graph_stats(data)
    
    # Generate visualizations
    visualize_graph(data, save_path="supply_chain_graph_visualization.png")
    visualize_graph_metrics(data, save_path="supply_chain_metrics.png")
    
    # Save the graph
    save_graph(data, "supply_chain_graph.pt")
    
    print("\n" + "="*50)
    print("✓ Graph construction complete and ready for GNN training!")
    print("="*50)
    print("\nGenerated files:")
    print("  1. supply_chain_graph.pt - PyTorch Geometric graph data")
    print("  2. supply_chain_graph_visualization.png - Graph structure visualization")
    print("  3. supply_chain_metrics.png - Feature distributions and metrics")
    print("\nYou can now use this graph with PyTorch Geometric GNN models.")
    print("Example usage:")
    print("  from graph_construction import load_graph")
    print("  data = load_graph('supply_chain_graph.pt')")
    print("  # Use data with your GNN model")
    print("="*50)
