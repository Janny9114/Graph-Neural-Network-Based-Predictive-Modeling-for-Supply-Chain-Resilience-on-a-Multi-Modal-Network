import torch

# Check multiple scenarios to find edge disruptions
for scenario_id in [0, 4, 5, 6, 7, 8]:  # Check edge-only scenarios
    data = torch.load(f'scenario_graphs_edge_disruptions/scenario_{scenario_id:05d}.pt', weights_only=False)
    
    print(f"\n{'='*70}")
    print(f"Scenario {scenario_id:05d}")
    print(f"{'='*70}")
    print(f"Scenario type: {data.scenario_type}")
    print(f"Disruption category: {data.disruption_category}")
    print(f"Edge attr shape: {data.edge_attr.shape}")
    
    # Find disrupted edges
    disrupted = [i for i in range(len(data.edge_attr)) if data.edge_attr[i, 4] > 0]
    print(f"Disrupted edges: {len(disrupted)}")
    
    if disrupted:
        print(f"\n✅ DYNAMIC EDGE FEATURES WORKING!")
        print(f"Sample disrupted edge: {data.edge_attr[disrupted[0]]}")
        print(f"\nDynamic features:")
        print(f"  [4] is_disrupted: {data.edge_attr[disrupted[0], 4]:.4f}")
        print(f"  [5] disruption_severity: {data.edge_attr[disrupted[0], 5]:.4f}")
        print(f"  [6] time_to_recovery: {data.edge_attr[disrupted[0], 6]:.4f}")
        print(f"\nModified static features (due to disruption):")
        print(f"  [0] lead_time (increased): {data.edge_attr[disrupted[0], 0]:.4f}")
        print(f"  [1] cost (increased 50%): {data.edge_attr[disrupted[0], 1]:.4f}")
        break
