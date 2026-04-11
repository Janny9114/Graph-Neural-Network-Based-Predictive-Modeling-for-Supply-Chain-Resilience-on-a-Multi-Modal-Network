"""
Enhanced script to map supply_chain_disruption_recovery.csv regions to synthetic nodes.
Adds 'synthetic_region', 'node_id', and ONE-HOT ENCODED categorical columns.
"""

import pandas as pd
import numpy as np

# Region mapping: External regions → Synthetic network regions
REGION_MAPPING = {
    'Asia-Pacific': ['China 1', 'China 2', 'India'],
    'Europe': ['Norway', 'Netherlands', 'England'],
    'North America': ['United States 1', 'Mexico'],
    'South America': ['Mexico'],
    'Africa/Middle East': ['Kenya']
}

def map_disruptions_to_nodes():
    """Map external disruption data to synthetic network nodes with one-hot encoding."""
    
    print("="*70)
    print("MAPPING DISRUPTIONS TO NODES WITH ONE-HOT ENCODING")
    print("="*70)
    
    # Load data
    print("\n📂 Loading data...")
    disruption_df = pd.read_csv('simulation_disruption/supply_chain_disruption_recovery.csv')
    nodes_df = pd.read_csv('synthetic_nodes.csv')
    
    print(f"  Loaded {len(disruption_df):,} disruption events")
    print(f"  Loaded {len(nodes_df):,} synthetic nodes")
    
    # Add synthetic_region column
    print("\n🗺️  Mapping regions...")
    disruption_df['synthetic_region'] = disruption_df['supplier_region'].apply(
        lambda x: np.random.choice(REGION_MAPPING.get(x, ['United States 1']))
    )
    
    # Map to node_id based on tier, region, AND size
    print("🔗 Assigning node IDs (matching tier, region, and size)...")
    
    # Create size mapping: supplier_size → capacity threshold
    SIZE_CAPACITY_THRESHOLDS = {
        'Large': (1000, float('inf')),   # capacity >= 1000
        'Medium': (700, 1000),            # 700 <= capacity < 1000
        'Small': (0, 700)                 # capacity < 700
    }
    
    def assign_node_id(row):
        tier = row['supplier_tier'] - 1  # Convert Tier 1→0, Tier 2→1, etc.
        if tier > 3:
            tier = 3
        
        region = row['synthetic_region']
        size = row['supplier_size']
        
        # Get capacity range for this size
        min_cap, max_cap = SIZE_CAPACITY_THRESHOLDS.get(size, (0, float('inf')))
        
        # Find matching nodes by tier, region, AND capacity (size)
        matching = nodes_df[
            (nodes_df['tier'] == tier) & 
            (nodes_df['region'] == region) &
            (nodes_df['capacity'] >= min_cap) &
            (nodes_df['capacity'] < max_cap)
        ]
        
        if len(matching) > 0:
            return matching.sample(1).index[0]
        else:
            # Fallback 1: Match tier and region only (ignore size)
            fallback1 = nodes_df[(nodes_df['tier'] == tier) & (nodes_df['region'] == region)]
            if len(fallback1) > 0:
                return fallback1.sample(1).index[0]
            else:
                # Fallback 2: Match tier only
                fallback2 = nodes_df[nodes_df['tier'] == tier]
                return fallback2.sample(1).index[0] if len(fallback2) > 0 else 0
    
    disruption_df['node_id'] = disruption_df.apply(assign_node_id, axis=1)
    
    # CONVERT SUPPLIER_TIER FROM 1-4 TO 0-3 (to match synthetic nodes)
    print("\n🔄 Converting supplier_tier from 1-4 to 0-3...")
    disruption_df['supplier_tier'] = disruption_df['supplier_tier'] - 1
    disruption_df['supplier_tier'] = disruption_df['supplier_tier'].clip(0, 3)
    print(f"  ✓ Tier range: {disruption_df['supplier_tier'].min()}-{disruption_df['supplier_tier'].max()}")
    
    # ONE-HOT ENCODE CATEGORICAL COLUMNS
    print("\n🔥 One-hot encoding categorical columns...")
    
    categorical_columns = [
        'disruption_type',
        'industry',
        'supplier_region',
        'supplier_size',
        'response_type'
    ]
    
    # Boolean columns (convert to 0/1)
    boolean_columns = ['has_backup_supplier', 'permanent_supplier_change']
    
    for col in boolean_columns:
        if col in disruption_df.columns:
            disruption_df[col] = disruption_df[col].astype(int)
            print(f"  ✓ Converted {col} to binary (0/1)")
    
    # One-hot encode categorical columns
    encoded_dfs = []
    for col in categorical_columns:
        if col in disruption_df.columns:
            # Create one-hot encoded columns
            one_hot = pd.get_dummies(disruption_df[col], prefix=col, dtype=int)
            encoded_dfs.append(one_hot)
            print(f"  ✓ One-hot encoded {col}: {len(one_hot.columns)} categories")
    
    # Combine original dataframe with one-hot encoded columns
    if encoded_dfs:
        encoded_df = pd.concat([disruption_df] + encoded_dfs, axis=1)
    else:
        encoded_df = disruption_df
    
    # DROP COLUMNS THAT CANNOT BE USED IN GNN
    print("\n🗑️  Dropping non-GNN compatible columns...")
    
    columns_to_drop = [
        # Original categorical columns (already one-hot encoded)
        'disruption_type',
        'industry',
        'supplier_region',
        'supplier_size',
        'response_type',
        # Text/ID columns
        'disruption_id',
        'synthetic_region'
    ]
    
    # Drop columns that exist
    columns_dropped = []
    for col in columns_to_drop:
        if col in encoded_df.columns:
            encoded_df = encoded_df.drop(columns=[col])
            columns_dropped.append(col)
            print(f"  ✓ Dropped: {col}")
    
    # Save mapped data in the same folder as the script
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, 'supply_chain_disruption_recovery_mapped.csv')
    encoded_df.to_csv(output_file, index=False)
    
    print("\n" + "="*70)
    print("✅ MAPPING COMPLETE!")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"  Total disruption events: {len(encoded_df):,}")
    print(f"  Unique nodes affected: {encoded_df['node_id'].nunique():,}")
    print(f"  Original columns: {len(disruption_df.columns)}")
    print(f"  Final columns (with one-hot): {len(encoded_df.columns)}")
    print(f"  Added one-hot columns: {len(encoded_df.columns) - len(disruption_df.columns)}")
    print(f"\n💾 Saved: {output_file}")
    
    # Show sample of one-hot encoded columns
    print("\n📋 Sample of one-hot encoded columns:")
    one_hot_cols = [col for col in encoded_df.columns if any(cat in col for cat in categorical_columns)]
    if one_hot_cols:
        print(f"  {', '.join(one_hot_cols[:10])}...")
    
    return encoded_df

if __name__ == "__main__":
    map_disruptions_to_nodes()
