import torch

# Load a training scenario (PyTorch 2.6+ requires weights_only=False for custom objects)
data = torch.load('scenario_graphs_edge_disruptions/scenario_00000.pt', weights_only=False)

print(f"Node features shape: {data.x.shape}")
# Should show: torch.Size([200, 16])

print(f"Feature breakdown:")
print(f"  Columns 0-5: Base features (capacity, cost, risk, reliability, x, y)")
print(f"  Columns 6-9: Tier one-hot encoding")
print(f"  Columns 10-13: ??? (duplicate tier or something else)")
print(f"  Columns 14-15: ??? (is_disrupted + severity)")