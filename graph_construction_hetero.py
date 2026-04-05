"""
Heterogeneous Graph Construction for Supply Chain GNN

This module creates PyTorch Geometric HeteroData objects with:
- Distinct node types: 'supplier', 'manufacturer', 'distributor', 'retailer'
- Distinct edge types: ('supplier', 'ships_to', 'manufacturer'), etc.
- Independent Z-score standardization for each node/edge type
"""

import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
from typing import Dict, Tuple
from graph_preprocessing import (
    calculate_node_flows,
    adjust_capacity_based_on_flows,
    verify_supply_chain_constraints,
    standardize_node_features_by_tier,
    preprocess_edge_flows_by_type
)


# Tier to node type mapping
TIER_TO_NODE_TYPE = {
    0: 'supplier',
    1: 'manufacturer',
    2: 'distributor',
    3: 'retailer'
}

# Edge type mapping
EDGE_TYPE_MAPPING = {
    (0, 1): ('supplier', 'ships_to', 'manufacturer'),
    (1, 2): ('manufacturer', 'ships_to', 'distributor'),
    (2, 3): ('distributor', 'ships_to', 'retailer')
}


def create_hetero_gnn_graph(
    node_path: str = "synthetic_nodes.csv",
    edge_path: str = "synthetic_edges.csv",
    adjust_capacity: bool = True,
    verify_constraints: bool = True,
    verbose: bool = True
) -> HeteroData:
    """
    Create a PyTorch Geometric HeteroData object from synthetic supply chain data.
    
    Features:
    - Distinct node types for each tier (supplier, manufacturer, distributor, retailer)
    - Distinct edge types for each connection
    - Independent Z-score standardization per node/edge type
    - Tier is NOT standardized (used as node type identifier)
    
    Args:
        node_path: Path to the nodes CSV file
        edge_path: Path to the edges CSV file
        adjust_capacity: Whether to adjust capacity based on flows
        verify_constraints: Whether to verify supply chain constraints
        verbose: Whether to print detailed information
        
    Returns:
        PyTorch Geometric HeteroData object
    """
    if verbose:
        print("="*70)
        print("HETEROGENEOUS SUPPLY CHAIN GRAPH CONSTRUCTION")
        print("="*70)
    
    # Load data
    node_df = pd.read_csv(node_path)
    edge_df = pd.read_csv(edge_path)
    
    if verbose:
        print(f"\n[1/6] Loaded {len(node_df)} nodes and {len(edge_df)} edges")
    
    # Calculate flows
    if verbose:
        print("\n[2/6] Calculating node flows...")
    node_df = calculate_node_flows(edge_df, node_df)
    
    # Adjust capacity
    if adjust_capacity:
        if verbose:
            print("\n[3/6] Adjusting capacity based on flows...")
        node_df = adjust_capacity_based_on_flows(node_df, buffer_factor=1.2)
    
    # Verify constraints
    if verify_constraints and verbose:
        print("\n[4/6] Verifying constraints...")
        verify_supply_chain_constraints(node_df, edge_df, verbose=verbose)
    
    # Standardize node features by tier
    if verbose:
        print("\n[5/6] Standardizing node features independently per tier...")
    
    feature_columns = ['capacity', 'cost_factor', 'risk_level', 'reliability', 'x', 'y']
    standardized_df, tier_feature_matrices = standardize_node_features_by_tier(
        node_df, feature_columns
    )
    
    if verbose:
        print(f"   ✓ Standardized {len(feature_columns)} features for each tier")
        for tier, features in tier_feature_matrices.items():
            node_type = TIER_TO_NODE_TYPE[tier]
            print(f"     - {node_type:15s}: {features.shape[0]} nodes, {features.shape[1]} features")
    
    # Preprocess edge flows by type
    if verbose:
        print("\n[6/6] Preprocessing edge flows independently per edge type...")
    
    edge_type_flows = preprocess_edge_flows_by_type(edge_df, node_df, 'flow_quantity')
    
    if verbose:
        print(f"   ✓ Preprocessed flows for {len(edge_type_flows)} edge types")
        for (src_tier, tgt_tier), flows in edge_type_flows.items():
            edge_type = EDGE_TYPE_MAPPING.get((src_tier, tgt_tier), f"tier_{src_tier}_to_{tgt_tier}")
            print(f"     - {str(edge_type):50s}: {len(flows)} edges")
    
    # Create HeteroData object
    if verbose:
        print("\n" + "="*70)
        print("Building HeteroData object...")
        print("="*70)
    
    data = HeteroData()
    
    # Add node features for each node type
    node_id_mapping = {}  # Maps original node_id to new index within node type
    
    for tier in sorted(node_df['tier'].unique()):
        node_type = TIER_TO_NODE_TYPE[tier]
        tier_mask = node_df['tier'] == tier
        tier_nodes = node_df[tier_mask]
        
        # Create mapping from original node_id to new index
        node_id_mapping[tier] = {
            original_id: new_idx 
            for new_idx, original_id in enumerate(tier_nodes.index)
        }
        
        # Add node features
        features = torch.tensor(tier_feature_matrices[tier], dtype=torch.float32)
        data[node_type].x = features
        data[node_type].num_nodes = len(tier_nodes)
        
        if verbose:
            print(f"\n{node_type.upper()} nodes:")
            print(f"  - Count: {data[node_type].num_nodes}")
            print(f"  - Features shape: {data[node_type].x.shape}")
            print(f"  - Feature mean: {data[node_type].x.mean(dim=0).tolist()}")
            print(f"  - Feature std: {data[node_type].x.std(dim=0).tolist()}")
    
    # Add edges for each edge type
    edge_df_with_tiers = edge_df.copy()
    edge_df_with_tiers['source_tier'] = edge_df['source'].map(node_df['tier'])
    edge_df_with_tiers['target_tier'] = edge_df['target'].map(node_df['tier'])
    
    for (src_tier, tgt_tier), group in edge_df_with_tiers.groupby(['source_tier', 'target_tier']):
        edge_type = EDGE_TYPE_MAPPING.get((src_tier, tgt_tier))
        
        if edge_type is None:
            if verbose:
                print(f"\nWarning: Unexpected edge type ({src_tier}, {tgt_tier}), skipping...")
            continue
        
        src_node_type, relation, tgt_node_type = edge_type
        
        # Map original node IDs to new indices within their node types
        src_indices = [node_id_mapping[src_tier][src_id] for src_id in group['source']]
        tgt_indices = [node_id_mapping[tgt_tier][tgt_id] for tgt_id in group['target']]
        
        # Create edge index
        edge_index = torch.tensor([src_indices, tgt_indices], dtype=torch.long)
        data[edge_type].edge_index = edge_index
        
        # Add edge features (standardized flows)
        flows = edge_type_flows[(src_tier, tgt_tier)]
        edge_attr = torch.tensor(flows, dtype=torch.float32).unsqueeze(1)
        data[edge_type].edge_attr = edge_attr
        
        if verbose:
            print(f"\n{edge_type} edges:")
            print(f"  - Count: {edge_index.shape[1]}")
            print(f"  - Edge index shape: {edge_index.shape}")
            print(f"  - Edge attr shape: {edge_attr.shape}")
            print(f"  - Flow mean: {edge_attr.mean().item():.6f}")
            print(f"  - Flow std: {edge_attr.std().item():.6f}")
    
    # Store metadata
    data.node_df = node_df
    data.edge_df = edge_df
    data.standardized_df = standardized_df
    data.node_id_mapping = node_id_mapping
    data.tier_to_node_type = TIER_TO_NODE_TYPE
    data.edge_type_mapping = EDGE_TYPE_MAPPING
    
    if verbose:
        print("\n" + "="*70)
        print("✓ HETEROGENEOUS GRAPH CONSTRUCTION COMPLETE!")
        print("="*70)
        print(f"\nNode types: {list(data.node_types)}")
        print(f"Edge types: {list(data.edge_types)}")
        print(f"\nTotal nodes: {sum(data[nt].num_nodes for nt in data.node_types)}")
        print(f"Total edges: {sum(data[et].edge_index.shape[1] for et in data.edge_types)}")
        print("="*70)
    
    return data


def save_hetero_graph(data: HeteroData, path: str = "supply_chain_hetero_graph.pt"):
    """Save the HeteroData object to disk."""
    torch.save(data, path)
    print(f"\n💾 Heterogeneous graph saved to: {path}")


def load_hetero_graph(path: str = "supply_chain_hetero_graph.pt") -> HeteroData:
    """Load a saved HeteroData object from disk."""
    data = torch.load(path)
    print(f"Heterogeneous graph loaded from {path}")
    return data


def print_hetero_graph_summary(data: HeteroData):
    """Print comprehensive summary of the HeteroData object."""
    print("\n" + "="*70)
    print("HETEROGENEOUS GRAPH SUMMARY")
    print("="*70)
    
    print(f"\n📊 Node Types: {len(data.node_types)}")
    for node_type in data.node_types:
        print(f"\n  {node_type.upper()}:")
        print(f"    - Count: {data[node_type].num_nodes}")
        print(f"    - Features: {data[node_type].x.shape}")
        print(f"    - Feature mean: {data[node_type].x.mean(dim=0).tolist()}")
        print(f"    - Feature std: {data[node_type].x.std(dim=0).tolist()}")
    
    print(f"\n🔗 Edge Types: {len(data.edge_types)}")
    for edge_type in data.edge_types:
        print(f"\n  {edge_type}:")
        print(f"    - Count: {data[edge_type].edge_index.shape[1]}")
        print(f"    - Edge index: {data[edge_type].edge_index.shape}")
        if hasattr(data[edge_type], 'edge_attr'):
            print(f"    - Edge attr: {data[edge_type].edge_attr.shape}")
            print(f"    - Flow mean: {data[edge_type].edge_attr.mean().item():.6f}")
            print(f"    - Flow std: {data[edge_type].edge_attr.std().item():.6f}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    print("Creating heterogeneous supply chain graph...\n")
    
    # Create HeteroData object
    data = create_hetero_gnn_graph(
        node_path="synthetic_nodes.csv",
        edge_path="synthetic_edges.csv",
        adjust_capacity=True,
        verify_constraints=True,
        verbose=True
    )
    
    # Print summary
    print_hetero_graph_summary(data)
    
    # Save the graph
    save_hetero_graph(data, "supply_chain_hetero_graph.pt")
    
    print("\n" + "="*70)
    print("✨ HETEROGENEOUS GRAPH GENERATION COMPLETE!")
    print("="*70)
    print("\nYou can now use this graph with PyTorch Geometric Heterogeneous GNN models.")
    print("\nExample usage:")
    print("```python")
    print("import torch")
    print("from torch_geometric.data import HeteroData")
    print("")
    print("# Load the graph")
    print("data = torch.load('supply_chain_hetero_graph.pt')")
    print("")
    print("# Access node features by type")
    print("supplier_features = data['supplier'].x")
    print("manufacturer_features = data['manufacturer'].x")
    print("")
    print("# Access edges by type")
    print("supplier_to_mfg_edges = data[('supplier', 'ships_to', 'manufacturer')].edge_index")
    print("supplier_to_mfg_flows = data[('supplier', 'ships_to', 'manufacturer')].edge_attr")
    print("```")
    print("="*70)
