"""
Map global supply chain disruption data to synthetic edges.
Maps disruptions to edges based on lead time similarity and destination region matching.
"""

import pandas as pd
import numpy as np
import os

def map_disruptions_to_edges():
    """Map disruption events to synthetic edges based on lead time and destination region."""
    
    print("="*70)
    print("MAPPING DISRUPTIONS TO SYNTHETIC EDGES")
    print("="*70)
    
    # Load data
    print("\n📂 Loading data...")
    disruption_df = pd.read_csv('simulation_disruption/global_supply_chain_disruption_v1.csv')
    edges_df = pd.read_csv('synthetic_edges.csv')
    nodes_df = pd.read_csv('synthetic_nodes.csv')
    
    print(f"  Loaded {len(disruption_df):,} disruption events")
    print(f"  Loaded {len(edges_df):,} synthetic edges")
    print(f"  Loaded {len(nodes_df):,} synthetic nodes")
    
    # FILTER OUT NONE DISRUPTION EVENTS
    print("\n🔍 Filtering disruption events...")
    print(f"  Before filtering: {len(disruption_df):,} events")
    
    disruption_df = disruption_df[
        (disruption_df['Disruption_Event'].notna()) & 
        (disruption_df['Disruption_Event'] != 'None')
    ]
    
    print(f"  After filtering: {len(disruption_df):,} events")
    print(f"  Removed: {len(pd.read_csv('simulation_disruption/global_supply_chain_disruption_v1.csv')) - len(disruption_df):,} None events")
    
    # Show disruption event distribution
    print("\n📊 Disruption Event Distribution:")
    event_counts = disruption_df['Disruption_Event'].value_counts()
    for event, count in event_counts.items():
        print(f"  {event}: {count:,} ({count/len(disruption_df)*100:.1f}%)")
    
    # MAP DESTINATION CITIES TO REGIONS
    print("\n🗺️  Mapping destination cities to regions...")
    
    # Create city to region mapping based on common cities
    city_to_region = {
        # China
        'Shanghai': 'China 1', 'Shenzhen': 'China 2', 'Beijing': 'China 1',
        'Guangzhou': 'China 2', 'Hong Kong': 'China 2',
        # US
        'Los Angeles': 'United States 1', 'New York': 'United States 1',
        'Seattle': 'United States 1', 'Chicago': 'United States 1',
        # Europe
        'Rotterdam': 'Netherlands', 'Hamburg': 'Netherlands', 
        'Felixstowe': 'England', 'London': 'England',
        'Oslo': 'Norway', 'Bergen': 'Norway',
        # Asia
        'Singapore': 'China 1', 'Tokyo': 'China 1', 'Mumbai': 'India',
        'Delhi': 'India', 'Bangalore': 'India',
        # Americas
        'Mexico City': 'Mexico', 'Monterrey': 'Mexico',
        'Santos': 'Mexico',  # Approximate
        # Africa
        'Nairobi': 'Kenya', 'Mombasa': 'Kenya'
    }
    
    def extract_city(city_str):
        """Extract city name from 'City, Country' format."""
        if pd.isna(city_str):
            return None
        return city_str.split(',')[0].strip()
    
    disruption_df['dest_city'] = disruption_df['Destination_City'].apply(extract_city)
    disruption_df['dest_region'] = disruption_df['dest_city'].map(city_to_region)
    
    # Fill unmapped cities with random region
    unmapped_mask = disruption_df['dest_region'].isna()
    if unmapped_mask.sum() > 0:
        disruption_df.loc[unmapped_mask, 'dest_region'] = np.random.choice(
            nodes_df['region'].unique(),
            size=unmapped_mask.sum()
        )
    
    print(f"  ✓ Mapped {(~unmapped_mask).sum()} cities to regions")
    print(f"  ✓ Assigned random regions to {unmapped_mask.sum()} unmapped cities")
    
    # MAP DISRUPTIONS TO EDGES
    print("\n🔗 Mapping disruptions to synthetic edges...")
    print("  Matching criteria:")
    print("    1. Similar lead time (±5 days)")
    print("    2. Matching destination region")
    
    # Add target region to edges
    edges_df['target_region'] = edges_df['target'].map(nodes_df['region'])
    
    edge_assignments = []
    matched_count = 0
    fallback_count = 0
    
    for idx, disruption in disruption_df.iterrows():
        lead_time = disruption['Base_Lead_Time_Days']
        dest_region = disruption['dest_region']
        
        # Find edges with similar lead time and matching destination region
        matching_edges = edges_df[
            (edges_df['lead_time'] >= lead_time - 5) &
            (edges_df['lead_time'] <= lead_time + 5) &
            (edges_df['target_region'] == dest_region)
        ]
        
        if len(matching_edges) > 0:
            # Randomly select one matching edge
            selected_edge = matching_edges.sample(1).iloc[0]
            edge_assignments.append({
                'source': selected_edge['source'],
                'target': selected_edge['target']
            })
            matched_count += 1
        else:
            # Fallback: match only on lead time
            fallback_edges = edges_df[
                (edges_df['lead_time'] >= lead_time - 5) &
                (edges_df['lead_time'] <= lead_time + 5)
            ]
            
            if len(fallback_edges) > 0:
                selected_edge = fallback_edges.sample(1).iloc[0]
                edge_assignments.append({
                    'source': selected_edge['source'],
                    'target': selected_edge['target']
                })
                fallback_count += 1
            else:
                # Last resort: random edge
                selected_edge = edges_df.sample(1).iloc[0]
                edge_assignments.append({
                    'source': selected_edge['source'],
                    'target': selected_edge['target']
                })
    
    # Add edge assignments to disruption dataframe
    edge_df_assignments = pd.DataFrame(edge_assignments)
    disruption_df['edge_source'] = edge_df_assignments['source'].values
    disruption_df['edge_target'] = edge_df_assignments['target'].values
    
    print(f"\n  ✓ Matched {matched_count} disruptions (lead time + region)")
    print(f"  ✓ Fallback matched {fallback_count} disruptions (lead time only)")
    print(f"  ✓ Random assigned {len(disruption_df) - matched_count - fallback_count} disruptions")
    
    # ONE-HOT ENCODE CATEGORICAL COLUMNS
    print("\n🔥 One-hot encoding categorical columns...")
    
    categorical_columns = [
        'Route_Type',
        'Transportation_Mode',
        'Product_Category',
        'Delivery_Status',
        'Disruption_Event',
        'Mitigation_Action_Taken'
    ]
    
    encoded_dfs = []
    for col in categorical_columns:
        if col in disruption_df.columns:
            one_hot = pd.get_dummies(disruption_df[col], prefix=col, dtype=int)
            encoded_dfs.append(one_hot)
            print(f"  ✓ One-hot encoded {col}: {len(one_hot.columns)} categories")
    
    # Combine original dataframe with one-hot encoded columns
    if encoded_dfs:
        encoded_df = pd.concat([disruption_df] + encoded_dfs, axis=1)
    else:
        encoded_df = disruption_df
    
    # DROP NON-GNN COMPATIBLE COLUMNS
    print("\n🗑️  Dropping non-GNN compatible columns...")
    
    columns_to_drop = [
        # Original categorical columns (already one-hot encoded)
        'Route_Type',
        'Transportation_Mode',
        'Product_Category',
        'Delivery_Status',
        'Disruption_Event',
        'Mitigation_Action_Taken',
        # Text/ID columns
        'Order_ID',
        'Order_Date',
        'Origin_City',
        'Destination_City',
        'dest_city',
        'dest_region'
    ]
    
    columns_dropped = []
    for col in columns_to_drop:
        if col in encoded_df.columns:
            encoded_df = encoded_df.drop(columns=[col])
            columns_dropped.append(col)
            print(f"  ✓ Dropped: {col}")
    
    # Save mapped data in the same folder as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, 'global_supply_chain_disruption_edge_mapped.csv')
    encoded_df.to_csv(output_file, index=False)
    
    print("\n" + "="*70)
    print("✅ EDGE MAPPING COMPLETE!")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"  Total disruption events (filtered): {len(encoded_df):,}")
    print(f"  Mapped to synthetic edges: {len(encoded_df):,}")
    print(f"  Original columns: {len(disruption_df.columns)}")
    print(f"  Final columns (with one-hot): {len(encoded_df.columns)}")
    print(f"  Edge columns added: edge_source, edge_target")
    print(f"\n💾 Saved: {output_file}")
    
    # Show sample of mapped data
    print("\n📋 Sample of mapped disruptions:")
    print(encoded_df[['edge_source', 'edge_target', 'Base_Lead_Time_Days', 'Delay_Days']].head())
    
    return encoded_df


if __name__ == "__main__":
    map_disruptions_to_edges()
