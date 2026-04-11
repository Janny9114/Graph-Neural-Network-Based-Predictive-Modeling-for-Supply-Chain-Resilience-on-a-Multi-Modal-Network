"""
Generate 1000 separate PyTorch Geometric Data objects for each disruption scenario.
Fixes the aggregation loss problem by preserving all scenario-specific labels.
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import os
from test_disruption_simulation import CascadingDisruptionSimulator

def generate_scenario_graphs():
    """Generate 1000 separate graph datasets, one per scenario."""
    
    print("="*70)
    print("GENERATING 1000 SCENARIO-SPECIFIC GRAPH DATASETS")
    print("="*70)
    
    # Load data
    print("\n📂 Loading data...")
    node_df = pd.read_csv('synthetic_nodes.csv')
    edge_df = pd.read_csv('synthetic_edges.csv')
    
    num_nodes = len(node_df)
    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {len(edge_df)}")
    
    # Create edge_index for PyG
    edge_index = torch.tensor([
        edge_df['source'].values,
        edge_df['target'].values
    ], dtype=torch.long)
    
    # Initialize simulator
    print("\n🎲 Initializing simulator...")
    simulator = CascadingDisruptionSimulator(
        resilience_threshold=0.6,
        propagation_decay=0.7,
        max_hops=3,
        seed=42
    )
    
    # Generate 1000 scenarios
    print("\n🔄 Generating 1000 cascading scenarios...")
    scenarios, centrality = simulator.generate_cascading_scenarios(
        node_df, edge_df,
        num_scenarios=1000,
        initial_disruption_prob=0.15
    )
    
    # Create output directory
    output_dir = 'scenario_graphs'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 Created output directory: {output_dir}/")
    
    # Process each scenario
    print("\n💾 Generating PyG Data objects for each scenario...")
    
    # Prepare base node features (static attributes)
    base_features = torch.tensor(
        node_df[['capacity', 'cost_factor', 'risk_level', 'reliability']].values,
        dtype=torch.float
    )
    num_base_features = base_features.shape[1]
    
    for scenario_idx, scenario in enumerate(scenarios):
        # Initialize feature matrix with zeros (baseline = no disruption)
        # Features: [capacity, cost_factor, risk_level, reliability, disruption_severity]
        x = torch.zeros((num_nodes, num_base_features + 1), dtype=torch.float)
        
        # Copy base features
        x[:, :num_base_features] = base_features
        
        # Initialize labels with -1 (unknown)
        y = torch.full((num_nodes,), -1, dtype=torch.long)
        
        # Initialize train mask (False for all)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        # Inject disruption features and labels for affected nodes
        for node_id in scenario['all_affected_nodes']:
            node_id = int(node_id)  # Convert to int
            severity = scenario['severities'][node_id]
            
            # Add disruption severity as feature
            x[node_id, num_base_features] = severity
            
            # Calculate label: resilient if severity < 0.5, vulnerable otherwise
            # (This is a simple threshold - you can make this more sophisticated)
            y[node_id] = 1 if severity < 0.5 else 0
            
            # Mark as training data
            train_mask[node_id] = True
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            scenario_id=scenario_idx,
            scenario_type=scenario['scenario_type'],
            initial_nodes=torch.tensor(scenario['initial_nodes'], dtype=torch.long),
            num_affected=len(scenario['all_affected_nodes'])
        )
        
        # Save to file
        output_path = os.path.join(output_dir, f'scenario_{scenario_idx:04d}.pt')
        torch.save(data, output_path)
        
        # Progress update every 100 scenarios
        if (scenario_idx + 1) % 100 == 0:
            print(f"  ✓ Generated {scenario_idx + 1}/1000 scenarios")
    
    print(f"\n✅ Successfully generated 1000 scenario graphs!")
    
    # Generate summary statistics
    print("\n" + "="*70)
    print("SCENARIO STATISTICS")
    print("="*70)
    
    total_affected = sum(len(s['all_affected_nodes']) for s in scenarios)
    total_initial = sum(len(s['initial_nodes']) for s in scenarios)
    
    print(f"\nOverall Statistics:")
    print(f"  Total scenarios: 1,000")
    print(f"  Avg initial disruptions per scenario: {total_initial / 1000:.1f}")
    print(f"  Avg total affected nodes per scenario: {total_affected / 1000:.1f}")
    print(f"  Avg propagation multiplier: {total_affected / total_initial:.2f}x")
    
    # Scenario type distribution
    scenario_types = {}
    for s in scenarios:
        stype = s['scenario_type']
        scenario_types[stype] = scenario_types.get(stype, 0) + 1
    
    print(f"\nScenario Type Distribution:")
    for stype, count in scenario_types.items():
        print(f"  {stype}: {count} ({count/10:.1f}%)")
    
    # Label distribution across all scenarios
    total_resilient = 0
    total_vulnerable = 0
    for scenario in scenarios:
        for node_id in scenario['all_affected_nodes']:
            severity = scenario['severities'][node_id]
            if severity < 0.5:
                total_resilient += 1
            else:
                total_vulnerable += 1
    
    total_labels = total_resilient + total_vulnerable
    print(f"\nLabel Distribution (across all scenarios):")
    print(f"  Resilient (1): {total_resilient:,} ({total_resilient/total_labels*100:.1f}%)")
    print(f"  Vulnerable (0): {total_vulnerable:,} ({total_vulnerable/total_labels*100:.1f}%)")
    
    print(f"\n💾 All scenario graphs saved to: {output_dir}/")
    print(f"   Files: scenario_0000.pt to scenario_0999.pt")
    
    # Create a metadata file
    metadata = {
        'num_scenarios': 1000,
        'num_nodes': num_nodes,
        'num_edges': len(edge_df),
        'num_features': num_base_features + 1,
        'feature_names': ['capacity', 'cost_factor', 'risk_level', 'reliability', 'disruption_severity'],
        'scenario_types': scenario_types,
        'total_resilient': total_resilient,
        'total_vulnerable': total_vulnerable
    }
    
    import json
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   Metadata: metadata.json")
    
    print("\n" + "="*70)
    print("✅ SCENARIO GRAPH GENERATION COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Use these 1000 graphs for GNN training")
    print("  2. Each graph has scenario-specific labels")
    print("  3. GNN will learn conditional resilience patterns")
    print("  4. No more aggregation loss!")


if __name__ == "__main__":
    generate_scenario_graphs()
