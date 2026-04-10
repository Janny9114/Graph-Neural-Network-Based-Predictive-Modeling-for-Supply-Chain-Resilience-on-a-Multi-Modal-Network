"""
Simple script to map supply_chain_disruption_recovery.csv regions to synthetic nodes.
Adds 'synthetic_region' and 'node_id' columns to link external data to your network.
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
    """Map external disruption data to synthetic network nodes."""
    
    # Load data
    disruption_df = pd.read_csv('supply_chain_disruption_recovery.csv')
    nodes_df = pd.read_csv('synthetic_nodes.csv')
    
    # Add synthetic_region column
    disruption_df['synthetic_region'] = disruption_df['supplier_region'].apply(
        lambda x: np.random.choice(REGION_MAPPING.get(x, ['United States 1']))
    )
    
    # Map to node_id based on tier and region
    def assign_node_id(row):
        tier = row['supplier_tier'] - 1  # Convert Tier 1→0, Tier 2→1, etc.
        if tier > 3:
            tier = 3
        
        region = row['synthetic_region']
        
        # Find matching nodes
        matching = nodes_df[(nodes_df['tier'] == tier) & (nodes_df['region'] == region)]
        
        if len(matching) > 0:
            return matching.sample(1).index[0]
        else:
            # Fallback: any node in that tier
            fallback = nodes_df[nodes_df['tier'] == tier]
            return fallback.sample(1).index[0] if len(fallback) > 0 else 0
    
    disruption_df['node_id'] = disruption_df.apply(assign_node_id, axis=1)
    
    # Save mapped data
    disruption_df.to_csv('supply_chain_disruption_recovery_mapped.csv', index=False)
    
    print(f"✅ Mapped {len(disruption_df):,} disruption events to {disruption_df['node_id'].nunique():,} nodes")
    print(f"💾 Saved: supply_chain_disruption_recovery_mapped.csv")

if __name__ == "__main__":
    map_disruptions_to_nodes()
