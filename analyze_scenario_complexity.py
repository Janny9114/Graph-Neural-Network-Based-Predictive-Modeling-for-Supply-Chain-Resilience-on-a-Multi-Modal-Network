"""
Analyze Scenario Complexity
Examines affected nodes in each scenario type to understand cascade patterns.
"""

import torch
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import os

def analyze_scenarios(scenario_dir='scenario_graphs_edge_disruptions'):
    """Analyze all scenarios and group by type."""
    
    # Load metadata
    metadata_path = os.path.join(scenario_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print("="*70)
    print("SCENARIO COMPLEXITY ANALYSIS")
    print("="*70)
    print(f"\nTotal scenarios: {metadata['num_scenarios']}")
    
    # Group scenarios by type
    scenario_stats = defaultdict(lambda: {
        'count': 0,
        'total_nodes': [],
        'affected_nodes': [],
        'failed_nodes': [],
        'degraded_nodes': [],
        'normal_nodes': [],
        'unlabeled_nodes': [],
        'disrupted_edges': [],
        'disrupted_node_count': []
    })
    
    # Analyze each scenario
    for i in range(metadata['num_scenarios']):
        scenario_path = os.path.join(scenario_dir, f'scenario_{i:05d}.pt')
        data = torch.load(scenario_path, weights_only=False)
        
        # Get scenario type
        scenario_type = data.scenario_type if hasattr(data, 'scenario_type') else 'unknown'
        
        # Count nodes by label
        labels = data.y.numpy()
        total_nodes = len(labels)
        
        failed = (labels == 0).sum()
        degraded = (labels == 1).sum()
        normal = (labels == 2).sum()
        unlabeled = (labels == -1).sum()
        affected = failed + degraded + normal  # Nodes with valid labels
        
        # Count disrupted edges
        num_edge_disruptions = data.num_edge_disruptions if hasattr(data, 'num_edge_disruptions') else 0
        
        # Count initially disrupted nodes
        num_node_disruptions = data.num_initial_disruptions if hasattr(data, 'num_initial_disruptions') else 0
        
        # Store stats
        stats = scenario_stats[scenario_type]
        stats['count'] += 1
        stats['total_nodes'].append(total_nodes)
        stats['affected_nodes'].append(affected)
        stats['failed_nodes'].append(failed)
        stats['degraded_nodes'].append(degraded)
        stats['normal_nodes'].append(normal)
        stats['unlabeled_nodes'].append(unlabeled)
        stats['disrupted_edges'].append(num_edge_disruptions)
        stats['disrupted_node_count'].append(num_node_disruptions)
    
    # Print summary for each scenario type
    print("\n" + "="*70)
    print("SCENARIO TYPE BREAKDOWN")
    print("="*70)
    
    results = []
    
    for scenario_type, stats in sorted(scenario_stats.items()):
        print(f"\n📊 {scenario_type.upper()}")
        print(f"   Count: {stats['count']} scenarios")
        
        # Calculate averages
        avg_affected = np.mean(stats['affected_nodes'])
        avg_failed = np.mean(stats['failed_nodes'])
        avg_degraded = np.mean(stats['degraded_nodes'])
        avg_normal = np.mean(stats['normal_nodes'])
        avg_unlabeled = np.mean(stats['unlabeled_nodes'])
        avg_edge_disruptions = np.mean(stats['disrupted_edges'])
        avg_node_disruptions = np.mean(stats['disrupted_node_count'])
        
        # Calculate percentages
        pct_failed = (avg_failed / avg_affected * 100) if avg_affected > 0 else 0
        pct_degraded = (avg_degraded / avg_affected * 100) if avg_affected > 0 else 0
        pct_normal = (avg_normal / avg_affected * 100) if avg_affected > 0 else 0
        pct_unlabeled = (avg_unlabeled / 200 * 100)  # Out of total 200 nodes
        
        print(f"\n   Average Affected Nodes: {avg_affected:.1f} / 200 ({avg_affected/200*100:.1f}%)")
        print(f"   ├─ Failed:    {avg_failed:.1f} ({pct_failed:.1f}%)")
        print(f"   ├─ Degraded:  {avg_degraded:.1f} ({pct_degraded:.1f}%)")
        print(f"   └─ Normal:    {avg_normal:.1f} ({pct_normal:.1f}%)")
        print(f"\n   Unlabeled Nodes: {avg_unlabeled:.1f} ({pct_unlabeled:.1f}%)")
        print(f"\n   Average Disruptions:")
        print(f"   ├─ Nodes: {avg_node_disruptions:.1f}")
        print(f"   └─ Edges: {avg_edge_disruptions:.1f}")
        
        # Calculate cascade ratio (affected nodes / initially disrupted)
        cascade_ratio = avg_affected / max(avg_node_disruptions, 1)
        print(f"\n   Cascade Ratio: {cascade_ratio:.2f}x")
        print(f"   (Each disrupted node affects {cascade_ratio:.1f} nodes on average)")
        
        results.append({
            'scenario_type': scenario_type,
            'count': stats['count'],
            'avg_affected': avg_affected,
            'avg_failed': avg_failed,
            'avg_degraded': avg_degraded,
            'avg_normal': avg_normal,
            'avg_unlabeled': avg_unlabeled,
            'pct_failed': pct_failed,
            'pct_degraded': pct_degraded,
            'pct_normal': pct_normal,
            'pct_unlabeled': pct_unlabeled,
            'avg_node_disruptions': avg_node_disruptions,
            'avg_edge_disruptions': avg_edge_disruptions,
            'cascade_ratio': cascade_ratio
        })
    
    # Create summary DataFrame
    df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print("\n" + df.to_string(index=False))
    
    # Save to CSV
    df.to_csv('scenario_complexity_analysis.csv', index=False)
    print(f"\n✓ Saved detailed analysis to: scenario_complexity_analysis.csv")
    
    # Overall statistics
    print("\n" + "="*70)
    print("OVERALL STATISTICS")
    print("="*70)
    
    total_scenarios = sum(stats['count'] for stats in scenario_stats.values())
    overall_avg_affected = df['avg_affected'].mean()
    overall_avg_failed = df['avg_failed'].mean()
    overall_avg_degraded = df['avg_degraded'].mean()
    overall_avg_normal = df['avg_normal'].mean()
    
    print(f"\nTotal Scenarios: {total_scenarios}")
    print(f"Average Affected Nodes: {overall_avg_affected:.1f} / 200 ({overall_avg_affected/200*100:.1f}%)")
    print(f"├─ Failed:   {overall_avg_failed:.1f} ({overall_avg_failed/overall_avg_affected*100:.1f}%)")
    print(f"├─ Degraded: {overall_avg_degraded:.1f} ({overall_avg_degraded/overall_avg_affected*100:.1f}%)")
    print(f"└─ Normal:   {overall_avg_normal:.1f} ({overall_avg_normal/overall_avg_affected*100:.1f}%)")
    
    # Identify most/least complex scenarios
    print("\n" + "="*70)
    print("COMPLEXITY RANKING")
    print("="*70)
    
    df_sorted = df.sort_values('cascade_ratio', ascending=False)
    
    print("\n🔥 Most Complex (Highest Cascade Ratio):")
    for idx, row in df_sorted.head(3).iterrows():
        print(f"   {row['scenario_type']}: {row['cascade_ratio']:.2f}x cascade")
    
    print("\n❄️  Least Complex (Lowest Cascade Ratio):")
    for idx, row in df_sorted.tail(3).iterrows():
        print(f"   {row['scenario_type']}: {row['cascade_ratio']:.2f}x cascade")
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE")
    print("="*70)
    
    return df


if __name__ == "__main__":
    df = analyze_scenarios()
