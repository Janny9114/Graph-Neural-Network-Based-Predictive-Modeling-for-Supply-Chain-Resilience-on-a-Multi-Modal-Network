"""Check if training data uses raw or normalized values."""
import torch
import pandas as pd

# Load one training scenario
data = torch.load('scenario_graphs_edge_disruptions/scenario_00000.pt', weights_only=False)

print(f"Feature shape: {data.x.shape}")
print(f"\nFirst node features:")
print(data.x[0])

print(f"\nFeature statistics across all nodes:")
print(f"Feature 0 (capacity): min={data.x[:, 0].min():.2f}, max={data.x[:, 0].max():.2f}, mean={data.x[:, 0].mean():.2f}, std={data.x[:, 0].std():.2f}")
print(f"Feature 1 (cost_factor): min={data.x[:, 1].min():.2f}, max={data.x[:, 1].max():.2f}, mean={data.x[:, 1].mean():.2f}, std={data.x[:, 1].std():.2f}")
print(f"Feature 2 (risk_level): min={data.x[:, 2].min():.2f}, max={data.x[:, 2].max():.2f}, mean={data.x[:, 2].mean():.2f}, std={data.x[:, 2].std():.2f}")
print(f"Feature 3 (reliability): min={data.x[:, 3].min():.2f}, max={data.x[:, 3].max():.2f}, mean={data.x[:, 3].mean():.2f}, std={data.x[:, 3].std():.2f}")
print(f"Feature 4 (x): min={data.x[:, 4].min():.2f}, max={data.x[:, 4].max():.2f}, mean={data.x[:, 4].mean():.2f}, std={data.x[:, 4].std():.2f}")
print(f"Feature 5 (y): min={data.x[:, 5].min():.2f}, max={data.x[:, 5].max():.2f}, mean={data.x[:, 5].mean():.2f}, std={data.x[:, 5].std():.2f}")
print(f"Feature 6-9 (tier one-hot): sum={data.x[:, 6:10].sum(dim=1).mean():.2f} (should be 1.0)")
print(f"Feature 10 (is_disrupted): min={data.x[:, 10].min():.2f}, max={data.x[:, 10].max():.2f}, sum={data.x[:, 10].sum():.0f}")

print(f"\n{'='*70}")
print("INTERPRETATION:")
print("="*70)
if data.x[:, 0].mean() > 100:
    print("✅ Features appear to be RAW values (capacity mean > 100)")
    print("   - capacity: large values (hundreds/thousands)")
    print("   - cost_factor, risk_level, reliability: 0-1 range")
elif abs(data.x[:, 0].mean()) < 0.1 and abs(data.x[:, 0].std() - 1.0) < 0.1:
    print("✅ Features appear to be NORMALIZED (mean ≈ 0, std ≈ 1)")
    print("   - All features have mean ≈ 0, std ≈ 1")
else:
    print("⚠️  Features are in an intermediate state")
    print(f"   - capacity mean: {data.x[:, 0].mean():.2f}")
    print(f"   - capacity std: {data.x[:, 0].std():.2f}")

# Load original node data to compare
print(f"\n{'='*70}")
print("COMPARING WITH ORIGINAL NODE DATA:")
print("="*70)
node_df = pd.read_csv('actual_pipeline_backup/synthetic_nodes.csv')
print(f"Original capacity: min={node_df['capacity'].min():.2f}, max={node_df['capacity'].max():.2f}, mean={node_df['capacity'].mean():.2f}")
print(f"Original cost_factor: min={node_df['cost_factor'].min():.2f}, max={node_df['cost_factor'].max():.2f}, mean={node_df['cost_factor'].mean():.2f}")
print(f"Original risk_level: min={node_df['risk_level'].min():.2f}, max={node_df['risk_level'].max():.2f}, mean={node_df['risk_level'].mean():.2f}")

if abs(data.x[0, 0].item() - node_df.iloc[0]['capacity']) < 1.0:
    print("\n✅ TRAINING DATA USES RAW VALUES (matches original data)")
else:
    print("\n✅ TRAINING DATA USES NORMALIZED VALUES (different from original)")
