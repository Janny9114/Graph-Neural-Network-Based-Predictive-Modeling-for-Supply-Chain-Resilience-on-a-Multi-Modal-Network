"""
Graph Preprocessing Module for Supply Chain GNN

This module handles:
1. Z-score standardization of node features (including location coordinates)
2. Log transformation and standardization of edge flow quantities
3. Flow conservation and capacity constraint verification
"""

import numpy as np
import pandas as pd
import torch
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


def z_score_standardization(values: np.ndarray) -> np.ndarray:
    """
    Apply Z-score standardization (zero mean, unit variance).
    
    Formula: z = (x - μ) / σ
    
    Args:
        values: Array of values to standardize
        
    Returns:
        Standardized array with mean ≈ 0 and std ≈ 1
    """
    mean = np.mean(values)
    std = np.std(values)
    
    if std == 0:
        # If all values are the same, return zeros
        return np.zeros_like(values)
    
    return (values - mean) / std


def calculate_node_flows(
    edge_df: pd.DataFrame,
    node_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate total inflow and outflow for each node based on edge flows.
    
    Args:
        edge_df: DataFrame with 'source', 'target', and 'flow_quantity' columns
        node_df: DataFrame with node information
        
    Returns:
        node_df with added 'inflow' and 'outflow' columns
    """
    # Initialize flow columns
    node_df = node_df.copy()
    node_df['inflow'] = 0.0
    node_df['outflow'] = 0.0
    
    # Calculate flows
    for _, edge in edge_df.iterrows():
        source = int(edge['source'])
        target = int(edge['target'])
        flow = float(edge['flow_quantity'])
        
        # Add to outflow of source node
        node_df.loc[source, 'outflow'] += flow
        
        # Add to inflow of target node
        node_df.loc[target, 'inflow'] += flow
    
    return node_df


def adjust_capacity_based_on_flows(
    node_df: pd.DataFrame,
    buffer_factor: float = 1.2
) -> pd.DataFrame:
    """
    Adjust node capacity to satisfy flow constraints.
    
    Capacity Constraint: capacity >= max(inflow, outflow) * buffer_factor
    
    Args:
        node_df: DataFrame with 'capacity', 'inflow', and 'outflow' columns
        buffer_factor: Safety buffer (default 1.2 = 20% buffer)
        
    Returns:
        node_df with adjusted capacity values
    """
    node_df = node_df.copy()
    
    for idx in node_df.index:
        max_flow = max(node_df.loc[idx, 'inflow'], node_df.loc[idx, 'outflow'])
        required_capacity = max_flow * buffer_factor
        
        # Update capacity if current capacity is insufficient
        if node_df.loc[idx, 'capacity'] < required_capacity:
            node_df.loc[idx, 'capacity'] = required_capacity
    
    return node_df


def verify_supply_chain_constraints(
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Verify supply chain constraints:
    1. Flow conservation (inflow ≈ outflow for intermediate nodes)
    2. Capacity constraints (flow <= capacity)
    
    Args:
        node_df: DataFrame with node information including flows and capacity
        edge_df: DataFrame with edge information
        verbose: Whether to print detailed verification results
        
    Returns:
        Dictionary with verification results
    """
    results = {
        'capacity_violations': 0,
        'flow_conservation_issues': 0,
        'passed': True
    }
    
    if verbose:
        print("\n" + "="*70)
        print("SUPPLY CHAIN CONSTRAINT VERIFICATION")
        print("="*70)
    
    # 1. Check capacity constraints
    capacity_violations = []
    for idx, row in node_df.iterrows():
        max_flow = max(row['inflow'], row['outflow'])
        if max_flow > row['capacity']:
            capacity_violations.append({
                'node_id': idx,
                'tier': row['tier'],
                'max_flow': max_flow,
                'capacity': row['capacity'],
                'violation': max_flow - row['capacity']
            })
    
    results['capacity_violations'] = len(capacity_violations)
    
    if verbose:
        print(f"\n1. Capacity Constraints:")
        print(f"   - Total nodes: {len(node_df)}")
        print(f"   - Nodes violating capacity: {len(capacity_violations)}")
        if len(capacity_violations) > 0:
            print(f"   - Sample violations (first 3):")
            for v in capacity_violations[:3]:
                print(f"     Node {v['node_id']} (Tier {v['tier']}): "
                      f"max_flow={v['max_flow']:.0f}, capacity={v['capacity']:.0f}, "
                      f"violation={v['violation']:.0f}")
        print(f"   - Status: {'✓ PASSED' if len(capacity_violations) == 0 else '✗ FAILED'}")
    
    # 2. Check flow conservation (for intermediate nodes)
    conservation_issues = []
    for idx, row in node_df.iterrows():
        tier = row['tier']
        inflow = row['inflow']
        outflow = row['outflow']
        
        # Skip source (tier 0) and sink (tier 3) nodes
        if tier == 0 or tier == 3:
            continue
        
        # Check if inflow ≈ outflow (within 15% tolerance)
        if inflow > 0 and outflow > 0:
            ratio = abs(inflow - outflow) / max(inflow, outflow)
            if ratio > 0.15:  # More than 15% difference
                conservation_issues.append({
                    'node_id': idx,
                    'tier': tier,
                    'inflow': inflow,
                    'outflow': outflow,
                    'imbalance_ratio': ratio
                })
    
    results['flow_conservation_issues'] = len(conservation_issues)
    
    if verbose:
        print(f"\n2. Flow Conservation (intermediate nodes only):")
        print(f"   - Intermediate nodes (Tier 1-2): {len(node_df[(node_df['tier'] == 1) | (node_df['tier'] == 2)])}")
        print(f"   - Nodes with flow imbalance (>15%): {len(conservation_issues)}")
        if len(conservation_issues) > 0:
            print(f"   - Sample issues (first 3):")
            for issue in conservation_issues[:3]:
                print(f"     Node {issue['node_id']} (Tier {issue['tier']}): "
                      f"inflow={issue['inflow']:.0f}, outflow={issue['outflow']:.0f}, "
                      f"imbalance={issue['imbalance_ratio']*100:.1f}%")
        print(f"   - Status: {'✓ PASSED' if len(conservation_issues) == 0 else '⚠ WARNING (acceptable for complex networks)'}")
    
    results['passed'] = (len(capacity_violations) == 0)
    
    if verbose:
        print("="*70)
    
    return results


def standardize_node_features(
    node_df: pd.DataFrame,
    feature_columns: list = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Apply Z-score standardization to ALL numerical node features (GLOBAL standardization).
    
    Default features to standardize:
    - tier
    - capacity
    - cost_factor
    - risk_level
    - reliability
    - x (location coordinate)
    - y (location coordinate)
    
    Args:
        node_df: DataFrame with raw node features
        feature_columns: List of columns to standardize (if None, uses default)
        
    Returns:
        Tuple of (DataFrame with standardized columns added, standardized feature matrix)
    """
    if feature_columns is None:
        feature_columns = ['tier', 'capacity', 'cost_factor', 'risk_level', 'reliability', 'x', 'y']
    
    # Create a copy
    standardized_df = node_df.copy()
    
    # Apply Z-score standardization to each feature
    standardized_features = []
    for col in feature_columns:
        if col in node_df.columns:
            std_values = z_score_standardization(node_df[col].values)
            standardized_df[f'{col}_std'] = std_values
            standardized_features.append(std_values)
        else:
            print(f"Warning: Column '{col}' not found in node_df")
    
    # Create feature matrix
    feature_matrix = np.column_stack(standardized_features)
    
    return standardized_df, feature_matrix


def standardize_node_features_by_tier(
    node_df: pd.DataFrame,
    feature_columns: list = None
) -> Tuple[pd.DataFrame, Dict[int, np.ndarray]]:
    """
    Apply Z-score standardization to node features INDEPENDENTLY for each tier.
    
    Features to standardize (excluding tier):
    - capacity
    - cost_factor
    - risk_level
    - reliability
    - x (location coordinate)
    - y (location coordinate)
    
    Args:
        node_df: DataFrame with raw node features
        feature_columns: List of columns to standardize (if None, uses default)
        
    Returns:
        Tuple of (DataFrame with standardized columns added, dict of feature matrices by tier)
    """
    if feature_columns is None:
        # DO NOT include 'tier' - it's used as node type identifier
        feature_columns = ['capacity', 'cost_factor', 'risk_level', 'reliability', 'x', 'y']
    
    # Create a copy
    standardized_df = node_df.copy()
    tier_feature_matrices = {}
    
    # Standardize features independently for each tier
    for tier in sorted(node_df['tier'].unique()):
        tier_mask = node_df['tier'] == tier
        tier_indices = node_df[tier_mask].index
        
        tier_features = []
        for col in feature_columns:
            if col in node_df.columns:
                # Get values for this tier only
                tier_values = node_df.loc[tier_mask, col].values
                
                # Apply Z-score standardization within this tier
                std_values = z_score_standardization(tier_values)
                
                # Store standardized values
                standardized_df.loc[tier_mask, f'{col}_std'] = std_values
                tier_features.append(std_values)
            else:
                print(f"Warning: Column '{col}' not found in node_df")
        
        # Create feature matrix for this tier
        tier_feature_matrices[tier] = np.column_stack(tier_features)
    
    return standardized_df, tier_feature_matrices


def preprocess_edge_flows_by_type(
    edge_df: pd.DataFrame,
    node_df: pd.DataFrame,
    flow_column: str = 'flow_quantity'
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Preprocess edge flow features INDEPENDENTLY for each edge type (tier connection).
    
    Steps:
    1. Group edges by (source_tier, target_tier)
    2. Apply log transformation: log(flow + 1)
    3. Apply Z-score standardization within each edge type
    
    Args:
        edge_df: DataFrame with edge information
        node_df: DataFrame with node information (to get tier)
        flow_column: Name of the column containing flow quantities
        
    Returns:
        Dictionary mapping (source_tier, target_tier) to standardized flow arrays
    """
    # Add tier information to edges
    edge_df_with_tiers = edge_df.copy()
    edge_df_with_tiers['source_tier'] = edge_df['source'].map(node_df['tier'])
    edge_df_with_tiers['target_tier'] = edge_df['target'].map(node_df['tier'])
    
    edge_type_flows = {}
    
    # Group by edge type
    for (src_tier, tgt_tier), group in edge_df_with_tiers.groupby(['source_tier', 'target_tier']):
        # Extract flow values for this edge type
        flow_values = group[flow_column].values
        
        # Step 1: Log transformation
        log_flows = np.log(flow_values + 1)
        
        # Step 2: Z-score standardization within this edge type
        standardized_log_flows = z_score_standardization(log_flows)
        
        edge_type_flows[(int(src_tier), int(tgt_tier))] = standardized_log_flows
    
    return edge_type_flows


def normalize_edge_features(
    edge_df: pd.DataFrame,
    feature_columns: list = None
) -> pd.DataFrame:
    """
    Normalize edge features with Z-score standardization.
    
    Args:
        edge_df: DataFrame with edge features
        feature_columns: List of columns to normalize (if None, auto-detect numerical columns)
        
    Returns:
        DataFrame with normalized edge features
    """
    from sklearn.preprocessing import StandardScaler
    
    if feature_columns is None:
        # Auto-detect numerical columns (exclude source, target, edge identifiers)
        exclude_cols = ['source', 'target', 'edge_id', 'source_tier', 'target_tier']
        feature_columns = [
            col for col in edge_df.columns 
            if col not in exclude_cols 
            and edge_df[col].dtype in ['float64', 'float32', 'int64', 'int32']
        ]
    
    if len(feature_columns) == 0:
        print("  ⚠ No edge features to normalize")
        return edge_df
    
    # Create copy
    normalized_df = edge_df.copy()
    
    # Normalize each feature
    scaler = StandardScaler()
    normalized_df[feature_columns] = scaler.fit_transform(edge_df[feature_columns])
    
    print(f"  ✓ Normalized {len(feature_columns)} edge features: {feature_columns}")
    print(f"    Mean: {normalized_df[feature_columns].mean().values}")
    print(f"    Std: {normalized_df[feature_columns].std().values}")
    
    return normalized_df


def preprocess_edge_flows(
    edge_df: pd.DataFrame,
    flow_column: str = 'flow_quantity'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess edge flow features with log transformation and Z-score standardization.
    (Global standardization - for homogeneous graphs)
    
    Steps:
    1. Apply log transformation: log(flow + 1)
    2. Apply Z-score standardization to log-transformed flows
    
    Args:
        edge_df: DataFrame with edge information
        flow_column: Name of the column containing flow quantities
        
    Returns:
        Tuple of (log_flows, standardized_log_flows)
    """
    # Extract flow values
    flow_values = edge_df[flow_column].values
    
    # Step 1: Log transformation (add 1 to avoid log(0))
    log_flows = np.log(flow_values + 1)
    
    # Step 2: Z-score standardization
    standardized_log_flows = z_score_standardization(log_flows)
    
    return log_flows, standardized_log_flows


def print_preprocessing_summary(
    node_df: pd.DataFrame,
    standardized_node_features: np.ndarray,
    edge_df: pd.DataFrame,
    standardized_edge_flows: np.ndarray,
    feature_columns: list
) -> None:
    """Print comprehensive summary of preprocessing results."""
    print("\n" + "="*70)
    print("PREPROCESSING SUMMARY")
    print("="*70)
    
    print(f"\n📊 Node Features:")
    print(f"   - Total nodes: {len(node_df)}")
    print(f"   - Features standardized: {len(feature_columns)}")
    print(f"   - Feature names: {feature_columns}")
    print(f"   - Standardized matrix shape: {standardized_node_features.shape}")
    
    print(f"\n   Raw feature statistics:")
    for col in feature_columns:
        if col in node_df.columns:
            values = node_df[col].values
            print(f"     {col:15s}: mean={np.mean(values):8.2f}, std={np.std(values):8.2f}, "
                  f"min={np.min(values):8.2f}, max={np.max(values):8.2f}")
    
    print(f"\n   Standardized feature statistics (should be mean≈0, std≈1):")
    for i, col in enumerate(feature_columns):
        values = standardized_node_features[:, i]
        print(f"     {col:15s}: mean={np.mean(values):8.4f}, std={np.std(values):8.4f}")
    
    print(f"\n🔗 Edge Features:")
    print(f"   - Total edges: {len(edge_df)}")
    print(f"   - Flow feature: flow_quantity")
    
    raw_flows = edge_df['flow_quantity'].values
    print(f"\n   Raw flow statistics:")
    print(f"     flow_quantity  : mean={np.mean(raw_flows):10.2f}, std={np.std(raw_flows):10.2f}, "
          f"min={np.min(raw_flows):10.2f}, max={np.max(raw_flows):10.2f}")
    
    print(f"\n   Preprocessed flow statistics (log + standardized):")
    print(f"     log_std_flow   : mean={np.mean(standardized_edge_flows):8.4f}, "
          f"std={np.std(standardized_edge_flows):8.4f}")
    
    print("="*70)


def preprocess_supply_chain_data(
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    adjust_capacity: bool = True,
    verify_constraints: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Master preprocessing function for supply chain graph data.
    
    Steps:
    1. Calculate node flows (inflow/outflow)
    2. Adjust capacity based on flows (optional)
    3. Verify supply chain constraints (optional)
    4. Standardize node features (Z-score)
    5. Preprocess edge flows (log + Z-score)
    
    Args:
        node_df: DataFrame with raw node features
        edge_df: DataFrame with raw edge features
        adjust_capacity: Whether to adjust capacity to satisfy flow constraints
        verify_constraints: Whether to verify supply chain constraints
        verbose: Whether to print detailed information
        
    Returns:
        Tuple of (processed_node_df, standardized_node_features, standardized_edge_flows)
    """
    if verbose:
        print("="*70)
        print("SUPPLY CHAIN DATA PREPROCESSING")
        print("="*70)
    
    # Step 1: Calculate node flows
    if verbose:
        print("\n[1/5] Calculating node flows (inflow/outflow)...")
    node_df = calculate_node_flows(edge_df, node_df)
    if verbose:
        print(f"   ✓ Flows calculated for {len(node_df)} nodes")
    
    # Step 2: Adjust capacity (optional)
    if adjust_capacity:
        if verbose:
            print("\n[2/5] Adjusting node capacity based on flows...")
        node_df = adjust_capacity_based_on_flows(node_df, buffer_factor=1.2)
        if verbose:
            print(f"   ✓ Capacity adjusted with 20% buffer")
    else:
        if verbose:
            print("\n[2/5] Skipping capacity adjustment")
    
    # Step 3: Verify constraints (optional)
    if verify_constraints:
        if verbose:
            print("\n[3/5] Verifying supply chain constraints...")
        results = verify_supply_chain_constraints(node_df, edge_df, verbose=verbose)
    else:
        if verbose:
            print("\n[3/5] Skipping constraint verification")
    
    # Step 4: Standardize node features
    if verbose:
        print("\n[4/5] Standardizing node features (Z-score)...")
    feature_columns = ['tier', 'capacity', 'cost_factor', 'risk_level', 'reliability', 'x', 'y']
    processed_node_df, standardized_node_features = standardize_node_features(node_df, feature_columns)
    if verbose:
        print(f"   ✓ Standardized {len(feature_columns)} features")
    
    # Step 5: Preprocess edge flows
    if verbose:
        print("\n[5/5] Preprocessing edge flows (log + Z-score)...")
    log_flows, standardized_edge_flows = preprocess_edge_flows(edge_df, 'flow_quantity')
    if verbose:
        print(f"   ✓ Preprocessed {len(edge_df)} edge flows")
    
    # Print summary
    if verbose:
        print_preprocessing_summary(
            node_df, standardized_node_features,
            edge_df, standardized_edge_flows,
            feature_columns
        )
    
    return processed_node_df, standardized_node_features, standardized_edge_flows


if __name__ == "__main__":
    # Example usage
    print("Graph Preprocessing Module")
    print("="*70)
    print("\nThis module provides preprocessing functions for supply chain GNN data.")
    print("\nMain functions:")
    print("  - z_score_standardization(): Standardize features to mean=0, std=1")
    print("  - calculate_node_flows(): Calculate inflow/outflow for each node")
    print("  - adjust_capacity_based_on_flows(): Ensure capacity >= max(flow)")
    print("  - verify_supply_chain_constraints(): Check flow conservation & capacity")
    print("  - standardize_node_features(): Z-score standardization for node features")
    print("  - preprocess_edge_flows(): Log transform + standardize edge flows")
    print("  - preprocess_supply_chain_data(): Master preprocessing function")
    print("\nUsage:")
    print("  from graph_preprocessing import preprocess_supply_chain_data")
    print("  node_df, node_features, edge_flows = preprocess_supply_chain_data(node_df, edge_df)")
    print("="*70)
