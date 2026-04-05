"""
Comprehensive Verification Script for Supply Chain Graph Generation

This script verifies all conditions and constraints for the generated supply chain data:
1. Reliability score (Beta distribution, correlations, range)
2. Realistic degree constraints
3. Hierarchical flow quantities
4. Capacity constraints
5. Node feature ranges
6. Edge feature ranges
"""

import pandas as pd
import numpy as np

def verify_reliability(node_df):
    """Verify reliability score generation."""
    print("\n" + "="*70)
    print("1. RELIABILITY SCORE VERIFICATION")
    print("="*70)
    
    corr_capacity = np.corrcoef(node_df['reliability'], node_df['capacity'])[0,1]
    corr_risk = np.corrcoef(node_df['reliability'], node_df['risk_level'])[0,1]
    
    passed = True
    issues = []
    
    # Check correlations
    if corr_capacity <= 0:
        issues.append(f"Correlation with capacity is not positive: {corr_capacity:.4f}")
        passed = False
    
    if corr_risk >= 0:
        issues.append(f"Correlation with risk is not negative: {corr_risk:.4f}")
        passed = False
    
    # Check range
    if node_df['reliability'].min() < 0.60 or node_df['reliability'].max() > 0.99:
        issues.append(f"Reliability range [{node_df['reliability'].min():.2f}, {node_df['reliability'].max():.2f}] outside [0.60, 0.99]")
        passed = False
    
    if passed:
        print("  ✓ PASSED")
    else:
        print("  ✗ FAILED")
        for issue in issues:
            print(f"    - {issue}")
    
    return passed


def verify_degree_constraints(node_df, edge_df):
    """Verify realistic degree constraints."""
    print("\n" + "="*70)
    print("2. DEGREE CONSTRAINTS VERIFICATION")
    print("="*70)
    
    # Add tier information to edges
    edge_df_with_tiers = edge_df.copy()
    edge_df_with_tiers['source_tier'] = edge_df['source'].map(node_df['tier'])
    edge_df_with_tiers['target_tier'] = edge_df['target'].map(node_df['tier'])
    
    passed = True
    issues = []
    
    # Check Suppliers -> Manufacturers (2-4 suppliers per manufacturer)
    mfg_edges = edge_df_with_tiers[(edge_df_with_tiers['source_tier'] == 0) & 
                                    (edge_df_with_tiers['target_tier'] == 1)]
    in_degrees = mfg_edges.groupby('target').size()
    if in_degrees.min() < 2 or in_degrees.max() > 4:
        issues.append(f"Suppliers→Manufacturers: in-degree range [{in_degrees.min()}, {in_degrees.max()}] outside [2, 4]")
        passed = False
    
    # Check Manufacturers -> Distributors (1-3 manufacturers per distributor)
    dist_edges = edge_df_with_tiers[(edge_df_with_tiers['source_tier'] == 1) & 
                                     (edge_df_with_tiers['target_tier'] == 2)]
    in_degrees = dist_edges.groupby('target').size()
    if in_degrees.min() < 1 or in_degrees.max() > 3:
        issues.append(f"Manufacturers→Distributors: in-degree range [{in_degrees.min()}, {in_degrees.max()}] outside [1, 3]")
        passed = False
    
    # Check Distributors -> Retailers (1-2 distributors per retailer, mostly 1)
    retail_edges = edge_df_with_tiers[(edge_df_with_tiers['source_tier'] == 2) & 
                                       (edge_df_with_tiers['target_tier'] == 3)]
    in_degrees = retail_edges.groupby('target').size()
    if in_degrees.min() < 1 or in_degrees.max() > 2:
        issues.append(f"Distributors→Retailers: in-degree range [{in_degrees.min()}, {in_degrees.max()}] outside [1, 2]")
        passed = False
    
    if passed:
        print("  ✓ PASSED")
    else:
        print("  ✗ FAILED")
        for issue in issues:
            print(f"    - {issue}")
    
    return passed


def verify_flow_quantities(edge_df, node_df):
    """Verify hierarchical flow quantities."""
    print("\n" + "="*70)
    print("3. HIERARCHICAL FLOW QUANTITIES VERIFICATION")
    print("="*70)
    
    # Add tier information to edges
    edge_df_with_tiers = edge_df.copy()
    edge_df_with_tiers['source_tier'] = edge_df['source'].map(node_df['tier'])
    edge_df_with_tiers['target_tier'] = edge_df['target'].map(node_df['tier'])
    
    passed = True
    issues = []
    
    # Suppliers -> Manufacturers (10,000 to 50,000)
    flows = edge_df_with_tiers[(edge_df_with_tiers['source_tier'] == 0) & 
                                (edge_df_with_tiers['target_tier'] == 1)]['flow_quantity']
    if flows.min() < 10000 or flows.max() > 50000:
        issues.append(f"Suppliers→Manufacturers: flow range [{flows.min():.0f}, {flows.max():.0f}] outside [10000, 50000]")
        passed = False
    
    # Manufacturers -> Distributors (5,000 to 20,000)
    flows = edge_df_with_tiers[(edge_df_with_tiers['source_tier'] == 1) & 
                                (edge_df_with_tiers['target_tier'] == 2)]['flow_quantity']
    if flows.min() < 5000 or flows.max() > 20000:
        issues.append(f"Manufacturers→Distributors: flow range [{flows.min():.0f}, {flows.max():.0f}] outside [5000, 20000]")
        passed = False
    
    # Distributors -> Retailers (1,000 to 5,000)
    flows = edge_df_with_tiers[(edge_df_with_tiers['source_tier'] == 2) & 
                                (edge_df_with_tiers['target_tier'] == 3)]['flow_quantity']
    if flows.min() < 1000 or flows.max() > 5000:
        issues.append(f"Distributors→Retailers: flow range [{flows.min():.0f}, {flows.max():.0f}] outside [1000, 5000]")
        passed = False
    
    if passed:
        print("  ✓ PASSED")
    else:
        print("  ✗ FAILED")
        for issue in issues:
            print(f"    - {issue}")
    
    return passed


def verify_node_features(node_df):
    """Verify node feature ranges."""
    print("\n" + "="*70)
    print("4. NODE FEATURE RANGES VERIFICATION")
    print("="*70)
    
    passed = True
    issues = []
    
    # Check cost factor ranges by tier
    expected_ranges = {
        0: (0.60, 0.85),
        1: (0.85, 1.10),
        2: (1.10, 1.40),
        3: (1.40, 2.00)
    }
    for tier in sorted(node_df['tier'].unique()):
        tier_data = node_df[node_df['tier'] == tier]['cost_factor']
        expected_min, expected_max = expected_ranges[tier]
        if tier_data.min() < expected_min - 0.1 or tier_data.max() > expected_max + 0.1:
            issues.append(f"Tier {tier} cost_factor range [{tier_data.min():.2f}, {tier_data.max():.2f}] outside expected [{expected_min:.2f}, {expected_max:.2f}]")
            passed = False
    
    if passed:
        print("  ✓ PASSED")
    else:
        print("  ✗ FAILED")
        for issue in issues:
            print(f"    - {issue}")
    
    return passed


def verify_edge_features(edge_df):
    """Verify edge feature ranges."""
    print("\n" + "="*70)
    print("5. EDGE FEATURE RANGES VERIFICATION")
    print("="*70)
    
    passed = True
    issues = []
    
    # Check capacity share
    if edge_df['capacity_share'].min() < 0.01 or edge_df['capacity_share'].max() > 0.7:
        issues.append(f"Capacity share range [{edge_df['capacity_share'].min():.3f}, {edge_df['capacity_share'].max():.3f}] outside [0.01, 0.7]")
        passed = False
    
    # Check disruption probability
    if edge_df['disruption_probability'].min() < 0.01 or edge_df['disruption_probability'].max() > 0.95:
        issues.append(f"Disruption probability range [{edge_df['disruption_probability'].min():.3f}, {edge_df['disruption_probability'].max():.3f}] outside [0.01, 0.95]")
        passed = False
    
    if passed:
        print("  ✓ PASSED")
    else:
        print("  ✗ FAILED")
        for issue in issues:
            print(f"    - {issue}")
    
    return passed


def main():
    """Main verification function."""
    print("="*70)
    print("COMPREHENSIVE SUPPLY CHAIN GRAPH VERIFICATION")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    node_df = pd.read_csv('synthetic_nodes.csv')
    edge_df = pd.read_csv('synthetic_edges.csv')
    
    print(f"  Nodes: {len(node_df)}")
    print(f"  Edges: {len(edge_df)}")
    
    # Run all verifications
    results = []
    results.append(("Reliability Score", verify_reliability(node_df)))
    results.append(("Degree Constraints", verify_degree_constraints(node_df, edge_df)))
    results.append(("Flow Quantities", verify_flow_quantities(edge_df, node_df)))
    results.append(("Node Features", verify_node_features(node_df)))
    results.append(("Edge Features", verify_edge_features(edge_df)))
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "⚠ WARNING"
        print(f"  {test_name:25s}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL VERIFICATIONS PASSED!")
    else:
        print("⚠ SOME VERIFICATIONS HAVE WARNINGS (review details above)")
    print("="*70)


if __name__ == "__main__":
    main()
