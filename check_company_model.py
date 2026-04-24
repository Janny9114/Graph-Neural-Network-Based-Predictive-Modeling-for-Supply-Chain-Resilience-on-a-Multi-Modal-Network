import torch

# Check company-specific model
company_id = 'sample_200_d4caf1f2'
model_path = f'backend/uploads/{company_id}/best_gine_model.pt'

checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

in_channels = checkpoint['conv1.nn.0.weight'].shape[1]
print(f"Company model ({company_id}) in_channels: {in_channels}")

# Also check training scenarios
scenario_path = f'backend/uploads/{company_id}/pipeline_scenarios/scenario_00000.pt'
try:
    data = torch.load(scenario_path, map_location='cpu', weights_only=False)
    print(f"Training scenario features: {data.x.shape[1]}")
    print(f"\n✅ Match: {in_channels == data.x.shape[1]}")
except:
    print(f"⚠️ Could not load training scenario")
