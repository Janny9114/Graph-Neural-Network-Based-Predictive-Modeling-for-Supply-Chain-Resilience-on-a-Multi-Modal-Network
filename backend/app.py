from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import torch
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import os
import uuid
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import your model class
import sys
sys.path.append('C:/Users/janny/Desktop/final_year')
from train_multi_gnn_realistic import GINEModel

# Load model
model = GINEModel(in_channels=11, edge_dim=4, hidden_channels=64, dropout=0.3, num_classes=3)
model.load_state_dict(torch.load('C:/Users/janny/Desktop/final_year/best_gine_model.pt', map_location=device))
model.to(device)
model.eval()

# Load graph data
node_df = pd.read_csv('C:/Users/janny/Desktop/final_year/synthetic_nodes.csv')
edge_df = pd.read_csv('C:/Users/janny/Desktop/final_year/synthetic_edges.csv')

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
        
        # Format response - Convert all numpy/pandas types to Python native types
        predictions = []
        for i in range(len(node_df)):
            predictions.append({
                'node_id': int(i),
                'node_name': str(node_df.iloc[i]['node_id']),
                'label': int(preds[i].item()),
                'label_name': ['Failed', 'Degraded', 'Normal'][int(preds[i].item())],
                'probability': [float(p) for p in probs[i].cpu().numpy().tolist()],
                'tier': int(node_df.iloc[i]['tier']),
                'region': str(node_df.iloc[i]['region'])
            })
        
        # Calculate summary - Convert to Python int
        summary = {
            'failed': int((preds == 0).sum().item()),
            'degraded': int((preds == 1).sum().item()),
            'normal': int((preds == 2).sum().item()),
            'total_nodes': int(len(node_df))
        }
        
        return jsonify({
            'predictions': predictions,
            'summary': summary,
            'status': 'success'
        })
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"ERROR in /api/predict: {error_traceback}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': error_traceback
        }), 500


def create_pyg_data(node_df, edge_df, disrupted_nodes, disrupted_edges, severity):
    """Create PyG Data object from disruption scenario."""
    num_nodes = len(node_df)
    
    # Node features (11 dimensions) - Make a COPY to avoid modifying original
    base_features = torch.tensor(
        node_df[['capacity', 'cost_factor', 'risk_level', 'reliability', 'x', 'y']].values,
        dtype=torch.float
    ).clone()
    
    # Apply severity to disrupted nodes - reduce capacity and reliability
    print(f"🔧 Applying severity {severity} to nodes {disrupted_nodes}")
    for node_id in disrupted_nodes:
        print(f"  Node {node_id} BEFORE: capacity={base_features[node_id, 0]:.3f}, reliability={base_features[node_id, 3]:.3f}, risk={base_features[node_id, 2]:.3f}")
        # Reduce capacity by severity amount
        base_features[node_id, 0] *= (1.0 - severity)  # capacity
        # Reduce reliability by severity amount
        base_features[node_id, 3] *= (1.0 - severity)  # reliability
        # Increase risk by severity amount
        base_features[node_id, 2] = min(1.0, base_features[node_id, 2] + severity)  # risk_level
        print(f"  Node {node_id} AFTER: capacity={base_features[node_id, 0]:.3f}, reliability={base_features[node_id, 3]:.3f}, risk={base_features[node_id, 2]:.3f}")
    
    # Tier one-hot encoding
    tier_encoding = torch.zeros((num_nodes, 4), dtype=torch.float)
    for idx, tier in enumerate(node_df['tier'].values):
        tier_encoding[idx, int(tier)] = 1.0
    
    # is_initially_disrupted
    is_disrupted = torch.zeros((num_nodes, 1), dtype=torch.float)
    for node_id in disrupted_nodes:
        is_disrupted[node_id, 0] = severity  # Use severity value instead of just 1.0
    
    x = torch.cat([base_features, tier_encoding, is_disrupted], dim=1)
    
    # Edge index - Convert to numpy array first to avoid warning
    edge_index = torch.tensor(
        np.array([edge_df['source'].values, edge_df['target'].values]),
        dtype=torch.long
    )
    
    # Edge features (4 dimensions)
    num_edges = len(edge_df)
    edge_attr = torch.zeros((num_edges, 4), dtype=torch.float)
    
    # Static edge features - use 'capacity_share' instead of 'weight'
    if 'weight' in edge_df.columns:
        weights = edge_df['weight'].values
    else:
        weights = edge_df['capacity_share'].values  # Use capacity_share as weight
    
    edge_attr[:, 0] = torch.tensor((weights - weights.mean()) / (weights.std() + 1e-8))
    edge_attr[:, 2] = torch.tensor(weights)
    
    # Add cost and disruption_prob
    for idx, row in edge_df.iterrows():
        source_idx = int(row['source'])
        target_idx = int(row['target'])
        
        source_cost = float(node_df.iloc[source_idx]['cost_factor'])
        target_cost = float(node_df.iloc[target_idx]['cost_factor'])
        edge_attr[idx, 1] = (source_cost + target_cost) / 2.0
        
        source_risk = float(node_df.iloc[source_idx]['risk_level'])
        target_risk = float(node_df.iloc[target_idx]['risk_level'])
        edge_attr[idx, 3] = (source_risk + target_risk) / 2.0
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


e@app.route('/api/health', methods=['GET'])
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
        # Use capacity_share if weight column doesn't exist
        weight_value = row.get('weight', row.get('capacity_share', 1.0))
        edges.append({
            'source': int(row['source']),
            'target': int(row['target']),
            'weight': float(weight_value)
        })
    
    return jsonify({
        'nodes': nodes,
        'edges': edges,
        'status': 'success'
    })


@app.route('/api/upload-graph', methods=['POST'])
def upload_graph():
    """Upload and validate company graph files."""
    
    # Check if files are present
    if 'nodes' not in request.files or 'edges' not in request.files:
        return jsonify({'status': 'error', 'message': 'Missing nodes or edges file'}), 400
    
    nodes_file = request.files['nodes']
    edges_file = request.files['edges']
    company_name = request.form.get('company_name', 'unknown')
    
    if nodes_file.filename == '' or edges_file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'}), 400
    
    if not (allowed_file(nodes_file.filename) and allowed_file(edges_file.filename)):
        return jsonify({'status': 'error', 'message': 'Only CSV files are allowed'}), 400
    
    # Generate company ID
    company_id = f"{company_name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
    
    # Create company directory
    company_dir = os.path.join(app.config['UPLOAD_FOLDER'], company_id)
    os.makedirs(company_dir, exist_ok=True)
    
    # Save files
    nodes_path = os.path.join(company_dir, 'nodes.csv')
    edges_path = os.path.join(company_dir, 'edges.csv')
    nodes_file.save(nodes_path)
    edges_file.save(edges_path)
    
    # Validate data
    try:
        validation_result = validate_graph_data(nodes_path, edges_path)
        
        if not validation_result['valid']:
            return jsonify({
                'status': 'error',
                'message': 'Validation failed',
                'errors': validation_result['errors']
            }), 400
        
        # Save metadata
        metadata = {
            'company_id': company_id,
            'company_name': company_name,
            'uploaded_at': datetime.now().isoformat(),
            'stats': validation_result['stats']
        }
        
        with open(os.path.join(company_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return jsonify({
            'status': 'success',
            'company_id': company_id,
            'stats': validation_result['stats']
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


def validate_graph_data(nodes_path, edges_path):
    """Validate uploaded graph data."""
    
    errors = []
    
    # Load data
    try:
        nodes_df = pd.read_csv(nodes_path)
        edges_df = pd.read_csv(edges_path)
    except Exception as e:
        return {'valid': False, 'errors': [f'Failed to read CSV: {str(e)}']}
    
    # Check required columns - match synthetic_nodes.csv and synthetic_edges.csv format
    required_node_cols = ['node_id', 'tier', 'capacity', 'cost_factor', 
                          'risk_level', 'reliability']
    # Note: x, y are used instead of latitude, longitude in synthetic data
    # Accept either format
    if 'x' not in nodes_df.columns and 'latitude' not in nodes_df.columns:
        errors.append('Missing location columns: need either (x, y) or (latitude, longitude)')
    if 'y' not in nodes_df.columns and 'longitude' not in nodes_df.columns:
        errors.append('Missing location columns: need either (x, y) or (latitude, longitude)')
    
    required_edge_cols = ['source', 'target']
    # capacity_share is optional - can use other weight columns
    
    missing_node_cols = set(required_node_cols) - set(nodes_df.columns)
    missing_edge_cols = set(required_edge_cols) - set(edges_df.columns)
    
    if missing_node_cols:
        errors.append(f'Missing node columns: {", ".join(missing_node_cols)}')
    if missing_edge_cols:
        errors.append(f'Missing edge columns: {", ".join(missing_edge_cols)}')
    
    # Normalize column names: latitude/longitude -> x/y
    if 'latitude' in nodes_df.columns and 'x' not in nodes_df.columns:
        nodes_df['x'] = nodes_df['longitude']
        nodes_df['y'] = nodes_df['latitude']
    
    # Add region if missing
    if 'region' not in nodes_df.columns:
        nodes_df['region'] = 'Unknown'
    
    # Add capacity_share to edges if missing (use equal weights)
    if 'capacity_share' not in edges_df.columns:
        if 'weight' in edges_df.columns:
            edges_df['capacity_share'] = edges_df['weight']
        else:
            edges_df['capacity_share'] = 1.0
    
    # Validate data types and ranges
    if (nodes_df['capacity'] <= 0).any():
        errors.append('Capacity must be positive')
    if (nodes_df['risk_level'] < 0).any() or (nodes_df['risk_level'] > 1).any():
        errors.append('Risk level must be between 0 and 1')
    if (nodes_df['reliability'] < 0).any() or (nodes_df['reliability'] > 1).any():
        errors.append('Reliability must be between 0 and 1')
    if (nodes_df['tier'] < 0).any() or (nodes_df['tier'] > 3).any():
        errors.append('Tier must be 0, 1, 2, or 3')
    
    # Check graph size
    num_nodes = len(nodes_df)
    if num_nodes < 20:
        errors.append(f'Graph too small ({num_nodes} nodes). Minimum 20 nodes required.')
    elif num_nodes < 50:
        errors.append(f'Warning: Small graph ({num_nodes} nodes). 50+ nodes recommended for better accuracy.')
    
    if errors:
        return {'valid': False, 'errors': errors}
    
    # Save normalized data back (only if validation passed)
    nodes_df.to_csv(nodes_path, index=False)
    edges_df.to_csv(edges_path, index=False)
    
    # Calculate stats
    stats = {
        'num_nodes': len(nodes_df),
        'num_edges': len(edges_df),
        'tiers': nodes_df['tier'].value_counts().to_dict(),
        'avg_capacity': float(nodes_df['capacity'].mean()),
        'avg_risk': float(nodes_df['risk_level'].mean())
    }
    
    return {'valid': True, 'errors': [], 'stats': stats}


@app.route('/api/generate-scenarios', methods=['POST'])
def generate_scenarios():
    """Start scenario generation and training task."""
    
    data = request.json
    company_id = data.get('company_id')
    num_scenarios = data.get('num_scenarios', 10000)
    epochs = data.get('epochs', 100)
    
    if not company_id:
        return jsonify({'status': 'error', 'message': 'Missing company_id'}), 400
    
    company_dir = os.path.join(app.config['UPLOAD_FOLDER'], company_id)
    if not os.path.exists(company_dir):
        return jsonify({'status': 'error', 'message': 'Company not found'}), 404
    
    # Generate a task ID
    task_id = uuid.uuid4().hex
    
    # Initialize task info
    task_info = {
        'task_id': task_id,
        'company_id': company_id,
        'num_scenarios': num_scenarios,
        'epochs': epochs,
        'status': 'started',
        'progress': 0,
        'message': 'Initializing training...',
        'stage': 'init',
        'started_at': datetime.now().isoformat()
    }
    
    task_path = os.path.join(company_dir, f'task_{task_id}.json')
    with open(task_path, 'w') as f:
        json.dump(task_info, f, indent=2)
    
    # Start training in a background thread
    import threading
    from training_pipeline import train_custom_graph
    
    def progress_callback(progress, message, stage):
        """Update task progress in JSON file."""
        task_info['progress'] = progress
        task_info['message'] = message
        task_info['stage'] = stage
        task_info['updated_at'] = datetime.now().isoformat()
        
        if progress >= 100:
            task_info['status'] = 'completed'
            task_info['completed_at'] = datetime.now().isoformat()
        
        with open(task_path, 'w') as f:
            json.dump(task_info, f, indent=2)
    
    def run_training():
        """Run training in background thread."""
        try:
            result = train_custom_graph(
                company_dir,
                num_scenarios=num_scenarios,
                epochs=epochs,
                progress_callback=progress_callback
            )
            
            # Update final status
            task_info['status'] = result['status']
            if result['status'] == 'success':
                task_info['model_path'] = result['model_path']
                task_info['metadata'] = result['metadata']
            else:
                task_info['error'] = result.get('message', 'Unknown error')
            
            with open(task_path, 'w') as f:
                json.dump(task_info, f, indent=2)
                
        except Exception as e:
            import traceback
            task_info['status'] = 'failed'
            task_info['error'] = str(e)
            task_info['traceback'] = traceback.format_exc()
            with open(task_path, 'w') as f:
                json.dump(task_info, f, indent=2)
    
    # Start background thread
    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()
    
    return jsonify({
        'status': 'started',
        'task_id': task_id,
        'message': f'Training started: {num_scenarios} scenarios, {epochs} epochs. This will take 2-4 hours.'
    })


@app.route('/api/task-status/<task_id>', methods=['GET'])
def task_status(task_id):
    """Get status of background task."""
    
    # Search for task file in all company directories
    for company_dir in os.listdir(app.config['UPLOAD_FOLDER']):
        task_path = os.path.join(app.config['UPLOAD_FOLDER'], company_dir, f'task_{task_id}.json')
        if os.path.exists(task_path):
            with open(task_path, 'r') as f:
                task_info = json.load(f)
            
            # Return actual progress from file
            return jsonify({
                'status': task_info.get('status', 'unknown'),
                'progress': task_info.get('progress', 0),
                'message': task_info.get('message', 'Processing...'),
                'stage': task_info.get('stage', ''),
                'started_at': task_info.get('started_at'),
                'updated_at': task_info.get('updated_at'),
                'model_path': task_info.get('model_path'),
                'error': task_info.get('error')
            })
    
    return jsonify({
        'status': 'not_found',
        'message': 'Task not found'
    }), 404


@app.route('/api/download-template/<template_type>', methods=['GET'])
def download_template(template_type):
    """Download CSV template."""
    
    if template_type == 'nodes':
        # Create nodes template
        template = pd.DataFrame({
            'node_id': [0, 1, 2, 3],
            'name': ['Supplier_A', 'Factory_X', 'Warehouse_Y', 'Store_Z'],
            'tier': [0, 1, 2, 3],
            'capacity': [1500, 2000, 1800, 500],
            'cost_factor': [0.75, 0.82, 0.71, 0.65],
            'risk_level': [0.35, 0.28, 0.31, 0.25],
            'reliability': [0.88, 0.91, 0.85, 0.93],
            'latitude': [40.7128, 35.6762, 34.0522, 41.8781],
            'longitude': [-74.0060, 139.6503, -118.2437, -87.6298],
            'region': ['North_America', 'Asia', 'North_America', 'North_America']
        })
        
        path = os.path.join(app.config['UPLOAD_FOLDER'], 'nodes_template.csv')
        template.to_csv(path, index=False)
        return send_file(path, as_attachment=True, download_name='nodes_template.csv')
    
    elif template_type == 'edges':
        # Create edges template
        template = pd.DataFrame({
            'source': [0, 1, 2],
            'target': [1, 2, 3],
            'capacity_share': [0.6, 0.8, 0.5],
            'lead_time': [7, 5, 2],
            'cost': [1500, 800, 300]
        })
        
        path = os.path.join(app.config['UPLOAD_FOLDER'], 'edges_template.csv')
        template.to_csv(path, index=False)
        return send_file(path, as_attachment=True, download_name='edges_template.csv')
    
    return jsonify({'status': 'error', 'message': 'Invalid template type'}), 400


if __name__ == '__main__':
    print("🚀 Starting GNN Backend API...")
    print(f"📊 Loaded {len(node_df)} nodes, {len(edge_df)} edges")
    print(f"🤖 Model: GINE (76.6% accuracy)")
    print(f"💻 Device: {device}")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    app.run(host='0.0.0.0', port=5000, debug=True)
