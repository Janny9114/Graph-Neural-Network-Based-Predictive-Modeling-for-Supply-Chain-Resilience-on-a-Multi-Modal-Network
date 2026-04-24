import torch
import os

# Check company-specific model
company_id = 'sample1_200_e35c04fa'
model_path = f'backend/uploads/{company_id}/best_gine_model.pt'

if not os.path.exists(model_path):
    print(f"❌ Model not found: {model_path}")
    print(f"\n📁 Available company directories:")
    uploads_dir = 'backend/uploads'
    if os.path.exists(uploads_dir):
        for company_dir in os.listdir(uploads_dir):
            print(f"   - {company_dir}")
    exit(1)

checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

# Try different key patterns for different model types
in_channels = None
if 'conv1.nn.0.weight' in checkpoint:
    # GINE/GIN model
    in_channels = checkpoint['conv1.nn.0.weight'].shape[1]
elif 'conv1.lin.weight' in checkpoint:
    # GAT/TransformerConv model
    in_channels = checkpoint['conv1.lin.weight'].shape[1]
elif 'conv1.weight' in checkpoint:
    # GCN/GraphSAGE model
    in_channels = checkpoint['conv1.weight'].shape[1]
else:
    print(f"⚠️ Unknown model architecture. Available keys:")
    for key in list(checkpoint.keys())[:10]:
        print(f"   - {key}")
    exit(1)

print(f"Company model ({company_id}) in_channels: {in_channels}")

# Also check training scenarios
scenario_path = f'backend/uploads/{company_id}/pipeline_scenarios/scenario_00000.pt'
try:
    data = torch.load(scenario_path, map_location='cpu', weights_only=False)
    print(f"Training scenario features: {data.x.shape[1]}")
    print(f"\n✅ Match: {in_channels == data.x.shape[1]}")
except Exception as e:
    print(f"⚠️ Could not load training scenario: {e}")
