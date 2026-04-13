from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torch_geometric.data import Data
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import your model class
from train_multi_gnn_realistic import GINEModel

# Load model
model = GINEModel(in_channels=11, edge_dim=4, hidden_channels=64, dropout=0.3, num_classes=3)
model.load_state_dict(torch.load('best_gine_model.pt', map_location=device))
model.to(device)
model.eval()

# Load graph data
node_df = pd.read_csv('synthetic_nodes.csv')
edge_df = pd.read_csv('synthetic_edges.csv')

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict node resilience given a disruption scenario.
    
    Request body:
    {
        "disrupted_nodes": [1, 5, 10],  # List of disrupted node IDs
        "disrupted_edges": [[1, 20], [5, 30]],  # List of disrupted edges
        "disruption_severity": 0.8  # 0.0-1.0
    }
    
    Response:
    {
        "predictions": [
            {"node_id": 0, "label": 2, "probability": [0.1, 0.2, 0.7]},
            {"node_id": 1, "label": 0, "probability": [0.8, 0.15, 0.05]},
            ...
        ],
        "summary": {
            "failed": 45,
            "degraded": 78,
            "normal": 77
        }
    }
    """
    try:
        data = request.json
        
        # Extract disruption info
        disrupted_nodes = data.get('disrupted_nodes', [])
        disrupted_edges = data.get('disrupted_edges', [])
        severity = data.get('disruption_severity', 0.8)
        
        # Create PyG Data object
        pyg_data = create_pyg_data(node_df, edge_df, disrupted_nodes, disrupted_edges, severity)
        pyg_data = pyg_data.to(device)
        
        # Make prediction
        with torch.no_grad():
            out = model(pyg_data.x, pyg_data.edge_index, pyg_data.edge_attr)
            probs = torch.exp(out)  # Convert log_softmax to probabilities
            preds = out.argmax(dim=1)
        
        # Format response
        predictions = []
        for i in range(len(node_df)):
            predictions.append({
                'node_id': int(i),
                'node_name': node_df.iloc[i]['node_id'],
                'label': int(preds[i]),
                'label_name': ['Failed', 'Degraded', 'Normal'][int(preds[i])],
                'probability': probs[i].cpu().numpy().tolist(),
                'tier': int(node_df.iloc[i]['tier']),
                'region': node_df.iloc[i]['region']
            })
        
        # Calculate summary
        summary = {
            'failed': int((preds == 0).sum()),
            'degraded': int((preds == 1).sum()),
            'normal': int((preds == 2).sum()),
            'total_nodes': len(node_df)
        }
        
        return jsonify({
            'predictions': predictions,
            'summary': summary,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


def create_pyg_data(node_df, edge_df, disrupted_nodes, disrupted_edges, severity):
    """Create PyG Data object from disruption scenario."""
    num_nodes = len(node_df)
    
    # Node features (11 dimensions)
    base_features = torch.tensor(
        node_df[['capacity', 'cost_factor', 'risk_level', 'reliability', 'x', 'y']].values,
        dtype=torch.float
    )
    
    # Tier one-hot encoding
    tier_encoding = torch.zeros((num_nodes, 4), dtype=torch.float)
    for idx, tier in enumerate(node_df['tier'].values):
        tier_encoding[idx, int(tier)] = 1.0
    
    # is_initially_disrupted
    is_disrupted = torch.zeros((num_nodes, 1), dtype=torch.float)
    for node_id in disrupted_nodes:
        is_disrupted[node_id, 0] = 1.0
    
    x = torch.cat([base_features, tier_encoding, is_disrupted], dim=1)
    
    # Edge index
    edge_index = torch.tensor([
        edge_df['source'].values,
        edge_df['target'].values
    ], dtype=torch.long)
    
    # Edge features (4 dimensions)
    num_edges = len(edge_df)
    edge_attr = torch.zeros((num_edges, 4), dtype=torch.float)
    
    # Static edge features
    weights = edge_df['weight'].values
    edge_attr[:, 0] = torch.tensor((weights - weights.mean()) / (weights.std() + 1e-8))
    edge_attr[:, 2] = torch.tensor(weights)
    
    # Add cost and disruption_prob
    for idx, row in edge_df.iterrows():
        source_cost = float(node_df.loc[row['source'], 'cost_factor'])
        target_cost = float(node_df.loc[row['target'], 'cost_factor'])
        edge_attr[idx, 1] = (source_cost + target_cost) / 2.0
        
        source_risk = float(node_df.loc[row['source'], 'risk_level'])
        target_risk = float(node_df.loc[row['target'], 'risk_level'])
        edge_attr[idx, 3] = (source_risk + target_risk) / 2.0
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'model': 'GINE', 'device': str(device)})


@app.route('/api/graph', methods=['GET'])
def get_graph():
    """Get graph structure for visualization."""
    nodes = []
    for idx, row in node_df.iterrows():
        nodes.append({
            'id': int(idx),
            'name': row['node_id'],
            'tier': int(row['tier']),
            'region': row['region'],
            'capacity': float(row['capacity']),
            'x': float(row['x']),
            'y': float(row['y'])
        })
    
    edges = []
    for idx, row in edge_df.iterrows():
        edges.append({
            'source': int(row['source']),
            'target': int(row['target']),
            'weight': float(row['weight'])
        })
    
    return jsonify({
        'nodes': nodes,
        'edges': edges,
        'status': 'success'
    })


if __name__ == '__main__':
    print("🚀 Starting GNN Backend API...")
    print(f"📊 Loaded {len(node_df)} nodes, {len(edge_df)} edges")
    print(f"🤖 Model: GINE (76.6% accuracy)")
    print(f"💻 Device: {device}")
    app.run(host='0.0.0.0', port=5000, debug=True)
