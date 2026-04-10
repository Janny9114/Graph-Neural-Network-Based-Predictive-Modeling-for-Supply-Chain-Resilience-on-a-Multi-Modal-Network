"""
View sample scenarios from the mixed targeted disruption simulation
"""

import pandas as pd
import numpy as np
import json

def view_scenarios():
    """Load and display sample scenarios."""
    print("="*70)
    print("SAMPLE SCENARIOS FROM MIXED TARGETED SIMULATION")
    print("="*70)
    
    # Load node data
    node_df = pd.read_csv('synthetic_nodes.csv')
    print(f"\nTotal nodes: {len(node_df)}")
    print(f"  Tier 0 (Suppliers): {len(node_df[node_df['tier'] == 0])}")
    print(f"  Tier 1 (Manufacturers): {len(node_df[node_df['tier'] == 1])}")
    print(f"  Tier 2 (Distributors): {len(node_df[node_df['tier'] == 2])}")
    print(f"  Tier 3 (Retailers): {len(node_df[node_df['tier'] == 3])}")
    
    # Load labels to see scenario impacts
    labels_df = pd.read_csv('node_resilience_labels_cascading.csv')
    print(f"\nTotal scenarios analyzed: 1000 (mixed targeted)")
    print(f"Nodes affected: {len(labels_df)}")
    print(f"  Resilient nodes: {(labels_df['resilient'] == 1).sum()} ({(labels_df['resilient'] == 1).sum() / len(labels_df) * 100:.1f}%)")
    print(f"  Vulnerable nodes: {(labels_df['resilient'] == 0).sum()} ({(labels_df['resilient'] == 0).sum() / len(labels_df) * 100:.1f}%)")
    
    # Show sample scenarios by type
    print("\n" + "="*70)
    print("SCENARIO TYPE EXAMPLES")
    print("="*70)
    
    # Example 1: Random Supplier Failure
    print("\n[SCENARIO TYPE 1] Random Supplier Failure")
    print("-" * 70)
    print("Description: 10 random suppliers fail simultaneously")
    print("Purpose: Test resilience to random disruptions")
    
    suppliers = node_df[node_df['tier'] == 0].sample(10, random_state=42)
    print(f"\nSample disrupted suppliers:")
    for idx, (node_id, row) in enumerate(suppliers.iterrows(), 1):
        print(f"  {idx}. Node {node_id}: {row['region']}, Capacity={row['capacity']:.0f}, Risk={row['risk_level']:.2f}")
    
    # Example 2: High-Capacity Supplier Failure
    print("\n[SCENARIO TYPE 2] High-Capacity Supplier Failure")
    print("-" * 70)
    print("Description: 5 highest-capacity suppliers fail")
    print("Purpose: Test impact of losing major suppliers")
    
    top_suppliers = node_df[node_df['tier'] == 0].nlargest(5, 'capacity')
    print(f"\nTop 5 capacity suppliers:")
    for idx, (node_id, row) in enumerate(top_suppliers.iterrows(), 1):
        print(f"  {idx}. Node {node_id}: {row['region']}, Capacity={row['capacity']:.0f}, Risk={row['risk_level']:.2f}")
    
    # Example 3: High-Risk Manufacturer Failure
    print("\n[SCENARIO TYPE 3] High-Risk Manufacturer Failure")
    print("-" * 70)
    print("Description: 8 highest-risk manufacturers fail")
    print("Purpose: Test impact of vulnerable manufacturers")
    
    high_risk_mfg = node_df[node_df['tier'] == 1].nlargest(8, 'risk_level')
    print(f"\nTop 8 high-risk manufacturers:")
    for idx, (node_id, row) in enumerate(high_risk_mfg.iterrows(), 1):
        print(f"  {idx}. Node {node_id}: {row['region']}, Capacity={row['capacity']:.0f}, Risk={row['risk_level']:.2f}")
    
    # Example 4: Central Distributor Failure
    print("\n[SCENARIO TYPE 4] Central Distributor Failure")
    print("-" * 70)
    print("Description: 4 most central distributors fail (highest betweenness)")
    print("Purpose: Test impact of losing critical network hubs")
    
    distributors = node_df[node_df['tier'] == 2].sample(4, random_state=42)
    print(f"\nSample central distributors:")
    for idx, (node_id, row) in enumerate(distributors.iterrows(), 1):
        print(f"  {idx}. Node {node_id}: {row['region']}, Capacity={row['capacity']:.0f}, Risk={row['risk_level']:.2f}")
    
    # Show cascading propagation example
    print("\n" + "="*70)
    print("CASCADING PROPAGATION EXAMPLE")
    print("="*70)
    
    print("\nScenario: Random Supplier Failure (Type 1)")
    print("-" * 70)
    
    # Simulate a scenario
    initial_nodes = suppliers.index.tolist()
    print(f"\nInitial disruption: {len(initial_nodes)} suppliers")
    print(f"  Severity: 70% reliability reduction (fixed)")
    
    # Load edge data to show propagation
    edge_df = pd.read_csv('synthetic_edges.csv')
    
    # Find downstream nodes (1-hop)
    downstream_1hop = edge_df[edge_df['source'].isin(initial_nodes)]['target'].unique()
    print(f"\nPropagation (1-hop downstream):")
    print(f"  Affected manufacturers: {len(downstream_1hop)} nodes")
    
    # Find 2-hop downstream
    downstream_2hop = edge_df[edge_df['source'].isin(downstream_1hop)]['target'].unique()
    print(f"\nPropagation (2-hop downstream):")
    print(f"  Affected distributors: {len(downstream_2hop)} nodes")
    
    # Find 3-hop downstream
    downstream_3hop = edge_df[edge_df['source'].isin(downstream_2hop)]['target'].unique()
    print(f"\nPropagation (3-hop downstream):")
    print(f"  Affected retailers: {len(downstream_3hop)} nodes")
    
    total_affected = len(initial_nodes) + len(downstream_1hop) + len(downstream_2hop) + len(downstream_3hop)
    print(f"\nTotal affected nodes: {total_affected}")
    print(f"Propagation multiplier: {total_affected / len(initial_nodes):.2f}x")
    
    # Show impact on resilience
    print("\n" + "="*70)
    print("IMPACT ON NODE RESILIENCE")
    print("="*70)
    
    # Sample some affected nodes
    sample_nodes = labels_df.sample(10, random_state=42)
    print("\nSample node resilience scores:")
    print(f"{'Node ID':<10} {'Disruptions':<15} {'Propagated':<15} {'Resilience':<12} {'Status'}")
    print("-" * 70)
    for _, row in sample_nodes.iterrows():
        status = "Resilient" if row['resilient'] == 1 else "Vulnerable"
        print(f"{int(row['node_id']):<10} {int(row['disruption_count']):<15} {int(row['propagated_count']):<15} {row['resilience_score']:.3f}        {status}")
    
    # Statistics
    print("\n" + "="*70)
    print("OVERALL STATISTICS")
    print("="*70)
    
    print(f"\nResilience Score Distribution:")
    print(f"  Mean: {labels_df['resilience_score'].mean():.3f}")
    print(f"  Std: {labels_df['resilience_score'].std():.3f}")
    print(f"  Min: {labels_df['resilience_score'].min():.3f}")
    print(f"  Max: {labels_df['resilience_score'].max():.3f}")
    
    print(f"\nDisruption Frequency:")
    print(f"  Avg disruptions per node: {labels_df['disruption_count'].mean():.1f}")
    print(f"  Avg propagated disruptions: {labels_df['propagated_count'].mean():.1f}")
    print(f"  Max disruptions: {labels_df['disruption_count'].max()}")
    
    print(f"\nNetwork Centrality Impact:")
    print(f"  Avg betweenness: {labels_df['betweenness'].mean():.4f}")
    print(f"  Avg PageRank: {labels_df['pagerank'].mean():.4f}")
    
    print("\n" + "="*70)
    print("✓ SCENARIO ANALYSIS COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    view_scenarios()
