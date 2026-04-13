"""
Analyze node and edge impact across different scenario types.
"""

import torch
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def analyze_scenarios(scenario_dir='scenario_graphs_edge_disruptions', num_to_analyze=10000):
    """Analyze impact statistics for different scenario types."""
    
    # Load metadata
    with open(f'{scenario_dir}/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"Analyzing {num_to_analyze} scenarios from {scenario_dir}...")
    print(f"Total scenarios available: {metadata['num_scenarios']}\n")
    
    # Statistics by scenario type
    stats_by_type = defaultdict(lambda: {
        'count': 0,
        'nodes_affected': [],
        'edges_disrupted': [],
        'class_distribution': defaultdict(int)
    })
    
    # Analyze scenarios
    for i in tqdm(range(min(num_to_analyze, metadata['num_scenarios'])), desc="Analyzing"):
        data = torch.load(f'{scenario_dir}/scenario_{i:05d}.pt', weights_only=False)
        
        scenario_type = data.scenario_type
        
        # Count affected nodes
        num_affected_nodes = data.train_mask.sum().item()
        
        # Count disrupted edges
        num_disrupted_edges = data.num_edge_disruptions if hasattr(data, 'num_edge_disruptions') else 0
        
        # Get class distribution
        labels = data.y[data.train_mask]
        for label in labels:
            stats_by_type[scenario_type]['class_distribution'][label.item()] += 1
        
        # Store statistics
        stats_by_type[scenario_type]['count'] += 1
        stats_by_type[scenario_type]['nodes_affected'].append(num_affected_nodes)
        stats_by_type[scenario_type]['edges_disrupted'].append(num_disrupted_edges)
    
    # Print results
    print("\n" + "="*80)
    print("SCENARIO IMPACT ANALYSIS")
    print("="*80)
    
    total_nodes = data.x.shape[0]
    total_edges = data.edge_index.shape[1]
    
    print(f"\nGraph Statistics:")
    print(f"  Total nodes: {total_nodes}")
    print(f"  Total edges: {total_edges}")
    
    print(f"\n{'='*80}")
    print("IMPACT BY SCENARIO TYPE")
    print(f"{'='*80}\n")
    
    for scenario_type, stats in sorted(stats_by_type.items()):
        print(f"📊 {scenario_type.upper().replace('_', ' ')}")
        print(f"   Count: {stats['count']} scenarios")
        
        # Node impact
        nodes_affected = np.array(stats['nodes_affected'])
        print(f"\n   Nodes Affected:")
        print(f"     Mean:   {nodes_affected.mean():.1f} ({nodes_affected.mean()/total_nodes*100:.1f}%)")
        print(f"     Median: {np.median(nodes_affected):.1f}")
        print(f"     Min:    {nodes_affected.min()}")
        print(f"     Max:    {nodes_affected.max()}")
        
        # Edge impact
        edges_disrupted = np.array(stats['edges_disrupted'])
        if edges_disrupted.sum() > 0:
            print(f"\n   Edges Disrupted:")
            print(f"     Mean:   {edges_disrupted.mean():.1f} ({edges_disrupted.mean()/total_edges*100:.1f}%)")
            print(f"     Median: {np.median(edges_disrupted):.1f}")
            print(f"     Min:    {edges_disrupted.min()}")
            print(f"     Max:    {edges_disrupted.max()}")
        else:
            print(f"\n   Edges Disrupted: None (node-only scenario)")
        
        # Class distribution
        print(f"\n   Class Distribution:")
        total_labels = sum(stats['class_distribution'].values())
        for label in sorted(stats['class_distribution'].keys()):
            count = stats['class_distribution'][label]
            pct = count / total_labels * 100
            class_name = {0: 'Failed', 1: 'Degraded', 2: 'Normal', 3: 'Normal'}.get(label, f'Class {label}')
            print(f"     Class {label} ({class_name}): {count} ({pct:.1f}%)")
        
        print(f"\n{'-'*80}\n")
    
    # Overall statistics
    print(f"{'='*80}")
    print("OVERALL STATISTICS")
    print(f"{'='*80}\n")
    
    all_nodes_affected = []
    all_edges_disrupted = []
    all_class_dist = defaultdict(int)
    
    for stats in stats_by_type.values():
        all_nodes_affected.extend(stats['nodes_affected'])
        all_edges_disrupted.extend(stats['edges_disrupted'])
        for label, count in stats['class_distribution'].items():
            all_class_dist[label] += count
    
    all_nodes_affected = np.array(all_nodes_affected)
    all_edges_disrupted = np.array(all_edges_disrupted)
    
    print(f"Nodes Affected (Average):")
    print(f"  Mean:   {all_nodes_affected.mean():.1f} ({all_nodes_affected.mean()/total_nodes*100:.1f}%)")
    print(f"  Median: {np.median(all_nodes_affected):.1f}")
    print(f"  Min:    {all_nodes_affected.min()}")
    print(f"  Max:    {all_nodes_affected.max()}")
    
    print(f"\nEdges Disrupted (Average):")
    print(f"  Mean:   {all_edges_disrupted.mean():.1f} ({all_edges_disrupted.mean()/total_edges*100:.1f}%)")
    print(f"  Median: {np.median(all_edges_disrupted):.1f}")
    print(f"  Min:    {all_edges_disrupted.min()}")
    print(f"  Max:    {all_edges_disrupted.max()}")
    
    print(f"\nOverall Class Distribution:")
    total_labels = sum(all_class_dist.values())
    for label in sorted(all_class_dist.keys()):
        count = all_class_dist[label]
        pct = count / total_labels * 100
        class_name = {0: 'Failed', 1: 'Degraded', 2: 'Normal', 3: 'Normal'}.get(label, f'Class {label}')
        print(f"  Class {label} ({class_name}): {count} ({pct:.1f}%)")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    analyze_scenarios('scenario_graphs_edge_disruptions', num_to_analyze=1000)
