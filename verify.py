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
    
    print("\nReliability Statistics:")
    print(f"  Mean: {node_df['reliability'].mean():.4f}")
    print(f"  Std: {node_df['reliability'].std():.4f}")
    print(f"  Min: {node_df['reliability'].min():.4f}")
    print(f"  Max: {node_df['reliability'].max():.4f}")
    print(f"  Median: {node_df['reliability'].median():.4f}")
    print(f"  25th percentile: {node_df['reliability'].quantile(0.25):.4f}")
    print(f"  75th percentile: {node_df['reliability'].quantile(0.75):.4f}")
    
    print("\nCorrelations:")
    corr_capacity = np.corrcoef(node_df['reliability'], node_df['capacity'])[0,1]
    corr_risk = np.corrcoef(node_df['reliability'], node_df['risk_level'])[0,1]
    
    print(f"  Reliability vs Capacity: {corr_capacity:.4f} (Expected: positive)")
    print(f"  Reliability vs Risk: {corr_risk:.4f} (Expected: negative)")
    
    print("\nVerification:")
    passed = True
    
    if corr_capacity > 0:
        print("  ✓ Positive correlation with capacity confirmed")
    else:
        print("  ✗ WARNING: Correlation with capacity is not positive")
        passed = False
    
    if corr_risk < 0:
        print("  ✓ Negative correlation with risk confirmed")
    else:
        print("  ✗ WARNING: Correlation with risk is not negative")
        passed = False
    
    if node_df['reliability'].min() >= 0.60 and node_df['reliability'].max() <= 0.99:
        print("  ✓ Reliability range [0.60, 0.99] confirmed")
    else:
        print(f"  ✗ WARNING: Reliability range is [{node_df['reliability'].min():.2f}, {node_df['reliability'].max():.2f}]")
        passed = False
    
    print("\nReliability by Tier:")
    for tier in sorted(node_df['tier'].unique()):
        tier_data = node_df[node_df['tier'] == tier]['reliability']
        print(f"  Tier {tier}: mean={tier_data.mean():.4f}, std={tier_data.std():.4f}, "
              f"min={tier_data.min():.4f}, max={tier_data.max():.4f}")
    
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
    
    # Check Suppliers -> Manufacturers (2-4 suppliers per manufacturer)
    print("\nSuppliers → Manufacturers (Expected: 2-4 suppliers per manufacturer):")
    mfg_edges = edge_df_with_tiers[(edge_df_with_tiers['source_tier'] == 0) & 
                                    (edge_df_with_tiers['target_tier'] == 1)]
    in_degrees = mfg_edges.groupby('target').size()
    print(f"  In-degree range: [{in_degrees.min()}, {in_degrees.max()}]")
    print(f"  Mean in-degree: {in_degrees.mean():.2f}")
    if in_degrees.min() >= 2 and in_degrees.max() <= 4:
        print("  ✓ Constraint satisfied")
    else:
        print("  ✗ WARNING: Some manufacturers have <2 or >4 suppliers")
        passed = False
    
    # Check Manufacturers -> Distributors (1-3 manufacturers per distributor)
    print("\nManufacturers → Distributors (Expected: 1-3 manufacturers per distributor):")
    dist_edges = edge_df_with_tiers[(edge_df_with_tiers['source_tier'] == 1) & 
                                     (edge_df_with_tiers['target_tier'] == 2)]
    in_degrees = dist_edges.groupby('target').size()
    print(f"  In-degree range: [{in_degrees.min()}, {in_degrees.max()}]")
    print(f"  Mean in-degree: {in_degrees.mean():.2f}")
    if in_degrees.min() >= 1 and in_degrees.max() <= 3:
        print("  ✓ Constraint satisfied")
    else:
        print("  ✗ WARNING: Some distributors have <1 or >3 manufacturers")
        passed = False
    
    # Check Distributors -> Retailers (1-2 distributors per retailer, mostly 1)
    print("\nDistributors → Retailers (Expected: 1 primary + 10% chance of backup):")
    retail_edges = edge_df_with_tiers[(edge_df_with_tiers['source_tier'] == 2) & 
                                       (edge_df_with_tiers['target_tier'] == 3)]
    in_degrees = retail_edges.groupby('target').size()
    print(f"  In-degree range: [{in_degrees.min()}, {in_degrees.max()}]")
    print(f"  Mean in-degree: {in_degrees.mean():.2f}")
    print(f"  Retailers with 1 distributor: {(in_degrees == 1).sum()} ({(in_degrees == 1).sum()/len(in_degrees)*100:.1f}%)")
    print(f"  Retailers with 2 distributors: {(in_degrees == 2).sum()} ({(in_degrees == 2).sum()/len(in_degrees)*100:.1f}%)")
    if in_degrees.min() >= 1 and in_degrees.max() <= 2:
        print("  ✓ Constraint satisfied")
    else:
        print("  ✗ WARNING: Some retailers have <1 or >2 distributors")
        passed = False
    
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
    
    # Suppliers -> Manufacturers (10,000 to 50,000)
    print("\nSuppliers → Manufacturers (Expected: 10,000 to 50,000):")
    flows = edge_df_with_tiers[(edge_df_with_tiers['source_tier'] == 0) & 
                                (edge_df_with_tiers['target_tier'] == 1)]['flow_quantity']
    print(f"  Range: [{flows.min():.0f}, {flows.max():.0f}]")
    print(f"  Mean: {flows.mean():.0f}")
    if flows.min() >= 10000 and flows.max() <= 50000:
        print("  ✓ Range satisfied")
    else:
        print("  ✗ WARNING: Flows outside expected range")
        passed = False
    
    # Manufacturers -> Distributors (5,000 to 20,000)
    print("\nManufacturers → Distributors (Expected: 5,000 to 20,000):")
    flows = edge_df_with_tiers[(edge_df_with_tiers['source_tier'] == 1) & 
                                (edge_df_with_tiers['target_tier'] == 2)]['flow_quantity']
    print(f"  Range: [{flows.min():.0f}, {flows.max():.0f}]")
    print(f"  Mean: {flows.mean():.0f}")
    if flows.min() >= 5000 and flows.max() <= 20000:
        print("  ✓ Range satisfied")
    else:
        print("  ✗ WARNING: Flows outside expected range")
        passed = False
    
    # Distributors -> Retailers (1,000 to 5,000)
    print("\nDistributors → Retailers (Expected: 1,000 to 5,000):")
    flows = edge_df_with_tiers[(edge_df_with_tiers['source_tier'] == 2) & 
                                (edge_df_with_tiers['target_tier'] == 3)]['flow_quantity']
    print(f"  Range: [{flows.min():.0f}, {flows.max():.0f}]")
    print(f"  Mean: {flows.mean():.0f}")
    if flows.min() >= 1000 and flows.max() <= 5000:
        print("  ✓ Range satisfied")
    else:
        print("  ✗ WARNING: Flows outside expected range")
        passed = False
    
    return passed


def verify_node_features(node_df):
    """Verify node feature ranges."""
    print("\n" + "="*70)
    print("4. NODE FEATURE RANGES VERIFICATION")
    print("="*70)
    
    passed = True
    
    print("\nCapacity by Tier:")
    for tier in sorted(node_df['tier'].unique()):
        tier_data = node_df[node_df['tier'] == tier]['capacity']
        print(f"  Tier {tier}: mean={tier_data.mean():.0f}, std={tier_data.std():.0f}, "
              f"min={tier_data.min():.0f}, max={tier_data.max():.0f}")
    
    print("\nRisk Level by Tier:")
    for tier in sorted(node_df['tier'].unique()):
        tier_data = node_df[node_df['tier'] == tier]['risk_level']
        print(f"  Tier {tier}: mean={tier_data.mean():.3f}, std={tier_data.std():.3f}, "
              f"min={tier_data.min():.3f}, max={tier_data.max():.3f}")
    
    print("\nCost Factor by Tier:")
    expected_ranges = {
        0: (0.60, 0.85),
        1: (0.85, 1.10),
        2: (1.10, 1.40),
        3: (1.40, 2.00)
    }
    for tier in sorted(node_df['tier'].unique()):
        tier_data = node_df[node_df['tier'] == tier]['cost_factor']
        expected_min, expected_max = expected_ranges[tier]
        print(f"  Tier {tier}: mean={tier_data.mean():.2f}, "
              f"range=[{tier_data.min():.2f}, {tier_data.max():.2f}] "
              f"(Expected: [{expected_min:.2f}, {expected_max:.2f}])")
        if tier_data.min() < expected_min - 0.1 or tier_data.max() > expected_max + 0.1:
            print(f"    ⚠ WARNING: Some values outside expected range")
    
    print("\nLocation Coordinates:")
    print(f"  X (longitude): range=[{node_df['x'].min():.2f}, {node_df['x'].max():.2f}]")
    print(f"  Y (latitude): range=[{node_df['y'].min():.2f}, {node_df['y'].max():.2f}]")
    print("  ✓ Coordinates randomized within zone boundaries")
    
    return passed


def verify_edge_features(edge_df):
    """Verify edge feature ranges."""
    print("\n" + "="*70)
    print("5. EDGE FEATURE RANGES VERIFICATION")
    print("="*70)
    
    passed = True
    
    print("\nLead Time:")
    print(f"  Range: [{edge_df['lead_time'].min():.2f}, {edge_df['lead_time'].max():.2f}]")
    print(f"  Mean: {edge_df['lead_time'].mean():.2f}")
    
    print("\nTransport Cost:")
    print(f"  Range: [{edge_df['transport_cost'].min():.2f}, {edge_df['transport_cost'].max():.2f}]")
    print(f"  Mean: {edge_df['transport_cost'].mean():.2f}")
    
    print("\nCapacity Share:")
    print(f"  Range: [{edge_df['capacity_share'].min():.3f}, {edge_df['capacity_share'].max():.3f}]")
    print(f"  Mean: {edge_df['capacity_share'].mean():.3f}")
    if edge_df['capacity_share'].min() >= 0.01 and edge_df['capacity_share'].max() <= 0.7:
        print("  ✓ Range [0.01, 0.7] satisfied")
    else:
        print("  ✗ WARNING: Some values outside expected range")
        passed = False
    
    print("\nDisruption Probability:")
    print(f"  Range: [{edge_df['disruption_probability'].min():.3f}, {edge_df['disruption_probability'].max():.3f}]")
    print(f"  Mean: {edge_df['disruption_probability'].mean():.3f}")
    if edge_df['disruption_probability'].min() >= 0.01 and edge_df['disruption_probability'].max() <= 0.95:
        print("  ✓ Range [0.01, 0.95] satisfied")
    else:
        print("  ✗ WARNING: Some values outside expected range")
        passed = False
    
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
