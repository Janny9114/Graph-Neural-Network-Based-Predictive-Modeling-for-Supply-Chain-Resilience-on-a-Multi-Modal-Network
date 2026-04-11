import pandas as pd
import numpy as np

def map_disruptions_to_edge():
    print("\n📂 Loading data...")
    disruption_df = pd.read_csv('global_supply_chain_disruption_v1.csv')
    edges_df = pd.read_csv('synthetic_edges.csv')

    print(f"  Loaded {len(disruption_df):,} disruption events")
    print(f"  Loaded {len(edges_df):,} synthetic edges")

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


if __name__ == "__main__":
    map_disruptions_to_edge()  
    
