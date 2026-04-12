# In graph_construction.py or graph_synthesis.py
import torch

# Load graph (need weights_only=False for PyG Data objects)
data = torch.load('supply_chain_graph.pt', weights_only=False)

print("="*70)
print("GRAPH FEATURE NORMALIZATION CHECK")
print("="*70)

# Check node features
print("\n📊 NODE FEATURES:")
print(f"Shape: {data.x.shape}")
print(f"  [num_nodes={data.x.shape[0]}, num_features={data.x.shape[1]}]")

# Node features structure:
# [0-3]: tier (one-hot, NOT normalized)
# [4-9]: capacity, cost_factor, risk_level, reliability, x, y (Z-score normalized)

print(f"\nTier features (one-hot, columns 0-3):")
tier_features = data.x[:, 0:4]
print(f"  Mean: {tier_features.mean(dim=0)}")
print(f"  Std: {tier_features.std(dim=0)}")
print(f"  ✓ One-hot encoded (not normalized, this is correct)")

print(f"\nNumerical features (Z-score, columns 4-9):")
print(f"  Features: [capacity, cost_factor, risk_level, reliability, x, y]")
numerical_features = data.x[:, 4:10]
print(f"  Mean: {numerical_features.mean(dim=0)}")
print(f"  Std: {numerical_features.std(dim=0)}")
print(f"  Min: {numerical_features.min(dim=0).values}")
print(f"  Max: {numerical_features.max(dim=0).values}")

# Check if normalized (mean≈0, std≈1)
mean_vals = numerical_features.mean(dim=0)
std_vals = numerical_features.std(dim=0)
is_normalized = (mean_vals.abs() < 0.01).all() and ((std_vals - 1.0).abs() < 0.1).all()

if is_normalized:
    print(f"  ✅ NORMALIZED! (mean≈0, std≈1)")
else:
    print(f"  ❌ NOT NORMALIZED! (mean should be ≈0, std should be ≈1)")

# Check edge features
print("\n🔗 EDGE FEATURES:")
if hasattr(data, 'edge_attr') and data.edge_attr is not None:
    print(f"Shape: {data.edge_attr.shape}")
    print(f"  [num_edges={data.edge_attr.shape[0]}, num_features={data.edge_attr.shape[1]}]")
    print(f"\nAll edge features (Z-score normalized):")
    print(f"  Features: [lead_time, transport_cost, capacity_share, disruption_probability]")
    print(f"  Mean: {data.edge_attr.mean(dim=0)}")
    print(f"  Std: {data.edge_attr.std(dim=0)}")
    print(f"  Min: {data.edge_attr.min(dim=0).values}")
    print(f"  Max: {data.edge_attr.max(dim=0).values}")
    
    # Check if normalized
    edge_mean = data.edge_attr.mean(dim=0)
    edge_std = data.edge_attr.std(dim=0)
    edge_normalized = (edge_mean.abs() < 0.01).all() and ((edge_std - 1.0).abs() < 0.1).all()
    
    if edge_normalized:
        print(f"  ✅ NORMALIZED! (mean≈0, std≈1)")
    else:
        print(f"  ❌ NOT NORMALIZED! (mean should be ≈0, std should be ≈1)")
else:
    print("No edge features (edge_attr is None)")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Node features: {'✅ NORMALIZED' if is_normalized else '❌ NOT NORMALIZED'}")
if hasattr(data, 'edge_attr') and data.edge_attr is not None:
    print(f"Edge features: {'✅ NORMALIZED' if edge_normalized else '❌ NOT NORMALIZED'}")
else:
    print(f"Edge features: ⚠ Not present")
print("="*70)
