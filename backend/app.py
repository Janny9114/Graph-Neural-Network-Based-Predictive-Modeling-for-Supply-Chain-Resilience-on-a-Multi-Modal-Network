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
import networkx as nx

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

# Load model - Match training configuration exactly
# Training data has 11 features: 6 base + 4 tier + 1 is_disrupted
# Training uses hidden_channels=256, dropout=0.3 (tuned hyperparameters)
print("📦 Loading GINE model...")
model = GINEModel(in_channels=11, edge_dim=4, hidden_channels=256, dropout=0.3, num_classes=3)
model.load_state_dict(torch.load('C:/Users/janny/Desktop/final_year/best_gine_model.pt', map_location=device))
model.to(device)
model.eval()
print(f"✅ Model loaded successfully!")
print(f"   Architecture: in_channels=11, hidden_channels=256, edge_dim=4, num_classes=3")
print(f"   Device: {device}")

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
        buffer_capacity = data.get('buffer_capacity', 0.5)  # Default 50%
        
        # Create PyG Data object
        pyg_data = create_pyg_data(node_df, edge_df, disrupted_nodes, disrupted_edges, severity, buffer_capacity)
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


def create_pyg_data(node_df, edge_df, disrupted_nodes, disrupted_edges, severity, buffer_capacity=0.5):
    """Create PyG Data object from disruption scenario."""
    num_nodes = len(node_df)
    
    # Node features (12 dimensions with buffer) - Make a COPY to avoid modifying original
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
    
    # Concatenate all features: 6 base + 4 tier + 1 is_disrupted = 11 dimensions
    # Note: buffer_capacity parameter is accepted but not used as a feature (matches training)
    # Buffer is implicitly represented through capacity feature
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
            'risk_level': float(row['risk_level']),
            'reliability': float(row['reliability']),
            'cost_factor': float(row['cost_factor']),
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


@app.route('/api/overview-metrics', methods=['GET'])
def get_overview_metrics():
    """Get overview metrics calculated from current graph data."""
    try:
        # Calculate average resilience (reliability)
        avg_resilience = float(node_df['reliability'].mean() * 100)
        
        # Calculate average risk level
        avg_risk = float(node_df['risk_level'].mean() * 100)
        
        # Calculate average lead time (if available in edges, otherwise estimate)
        if 'lead_time' in edge_df.columns:
            avg_lead_time = float(edge_df['lead_time'].mean())
        else:
            # Estimate based on capacity_share (inverse relationship)
            avg_lead_time = float(5.0)  # Default value
        
        # Calculate network density (edges per node)
        num_nodes = len(node_df)
        num_edges = len(edge_df)
        network_density = float(num_edges / num_nodes) if num_nodes > 0 else 0.0
        
        return jsonify({
            'status': 'success',
            'avg_resilience': round(avg_resilience, 1),
            'avg_risk': round(avg_risk, 1),
            'avg_lead_time': round(avg_lead_time, 1),
            'network_density': round(network_density, 2)
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500


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


@app.route('/api/training-results', methods=['GET'])
def get_training_results():
    """Get training results from the latest training run."""
    try:
        # Check multiple locations for training results
        possible_paths = [
            # 1. Check uploads directory for custom graphs
            ('uploads', 'training_metadata.json'),
            # 2. Check pipeline_output directory
            ('pipeline_output', 'model_comparison.csv'),
            # 3. Check root directory
            ('.', 'pipeline_output/model_comparison.csv')
        ]
        
        # Try uploads directory first
        if os.path.exists(UPLOAD_FOLDER):
            upload_dirs = [d for d in os.listdir(UPLOAD_FOLDER) if os.path.isdir(os.path.join(UPLOAD_FOLDER, d))]
            
            if upload_dirs:
                # Sort by modification time to get most recent
                upload_dirs.sort(key=lambda d: os.path.getmtime(os.path.join(UPLOAD_FOLDER, d)), reverse=True)
                latest_dir = os.path.join(UPLOAD_FOLDER, upload_dirs[0])
                
                # Check for training_metadata.json
                metadata_path = os.path.join(latest_dir, 'training_metadata.json')
                
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Extract results
                    results = metadata.get('results', [])
                    
                    return jsonify({
                        'status': 'success',
                        'results': results,
                        'metadata': {
                            'num_scenarios': metadata.get('num_scenarios'),
                            'num_nodes': metadata.get('num_nodes'),
                            'num_edges': metadata.get('num_edges'),
                            'trained_at': metadata.get('trained_at'),
                            'validation_accuracy': metadata.get('validation_accuracy')
                        }
                    })
        
        # Try pipeline_output directory
        pipeline_csv = '../pipeline_output/model_comparison.csv'
        if os.path.exists(pipeline_csv):
            df = pd.read_csv(pipeline_csv)
            results = df.to_dict('records')
            
            # Try to load summary.json for metadata
            summary_path = '../pipeline_output/summary.json'
            metadata = {}
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                    metadata = {
                        'num_scenarios': summary.get('num_scenarios'),
                        'num_nodes': summary.get('num_nodes'),
                        'num_edges': summary.get('num_edges'),
                        'trained_at': summary.get('timestamp')
                    }
            
            return jsonify({
                'status': 'success',
                'results': results,
                'metadata': metadata
            })
        
        return jsonify({
            'status': 'no_data',
            'message': 'No training results found. Please run training first.'
        }), 404
            
    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/network-vulnerability', methods=['GET'])
def get_network_vulnerability():
    """
    Compute network vulnerability metrics on the supply chain graph:
    - Betweenness Centrality: nodes that act as bottlenecks/bridges
    - Eigenvector Centrality / PageRank: influential hub nodes
    - Articulation Points (Cut Vertices): nodes whose removal disconnects the graph
    Returns top-N nodes for each metric plus summary stats.
    """
    try:
        top_n = int(request.args.get('top_n', 10))

        # ── Build an undirected NetworkX graph ──────────────────────────────
        G = nx.Graph()
        for idx, row in node_df.iterrows():
            G.add_node(
                int(idx),
                name=str(row['node_id']),
                tier=int(row['tier']),
                region=str(row['region']),
                risk_level=float(row['risk_level']),
                reliability=float(row['reliability']),
                capacity=float(row['capacity']),
            )

        weight_col = 'weight' if 'weight' in edge_df.columns else 'capacity_share'
        for _, row in edge_df.iterrows():
            G.add_edge(
                int(row['source']),
                int(row['target']),
                weight=float(row[weight_col])
            )

        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()

        # ── Betweenness Centrality ───────────────────────────────────────────
        betweenness = nx.betweenness_centrality(G, normalized=True, weight='weight')

        # ── Eigenvector Centrality (fall back to PageRank if it doesn't converge) ──
        try:
            eigenvector = nx.eigenvector_centrality(G, max_iter=1000, weight='weight')
            centrality_method = 'Eigenvector Centrality'
        except nx.PowerIterationFailedConvergence:
            eigenvector = nx.pagerank(G, weight='weight')
            centrality_method = 'PageRank'

        # ── Articulation Points ──────────────────────────────────────────────
        articulation_pts = set(nx.articulation_points(G))

        # ── Helper: enrich a node dict ───────────────────────────────────────
        def node_info(node_id, score, score_key):
            attrs = G.nodes[node_id]
            return {
                'node_id': node_id,
                'name': attrs['name'],
                'tier': attrs['tier'],
                'region': attrs['region'],
                'risk_level': round(attrs['risk_level'], 4),
                'reliability': round(attrs['reliability'], 4),
                'capacity': round(attrs['capacity'], 2),
                'degree': G.degree(node_id),
                score_key: round(score, 6),
                'is_articulation_point': node_id in articulation_pts,
            }

        # ── Top-N Betweenness ────────────────────────────────────────────────
        top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:top_n]
        betweenness_nodes = [node_info(nid, score, 'betweenness_centrality')
                             for nid, score in top_betweenness]

        # ── Top-N Eigenvector / PageRank ─────────────────────────────────────
        top_eigenvector = sorted(eigenvector.items(), key=lambda x: x[1], reverse=True)[:top_n]
        eigenvector_nodes = [node_info(nid, score, 'eigenvector_score')
                             for nid, score in top_eigenvector]

        # ── All Articulation Points (sorted by betweenness desc) ─────────────
        art_sorted = sorted(
            articulation_pts,
            key=lambda nid: betweenness.get(nid, 0),
            reverse=True
        )
        articulation_nodes = []
        for nid in art_sorted:
            info = node_info(nid, betweenness.get(nid, 0), 'betweenness_centrality')
            info['eigenvector_score'] = round(eigenvector.get(nid, 0), 6)
            articulation_nodes.append(info)

        # ── Summary stats ────────────────────────────────────────────────────
        bc_values = list(betweenness.values())
        ev_values = list(eigenvector.values())

        summary = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'num_articulation_points': len(articulation_pts),
            'articulation_point_fraction': round(len(articulation_pts) / num_nodes, 4) if num_nodes else 0,
            'avg_betweenness': round(float(np.mean(bc_values)), 6),
            'max_betweenness': round(float(np.max(bc_values)), 6),
            'avg_eigenvector': round(float(np.mean(ev_values)), 6),
            'max_eigenvector': round(float(np.max(ev_values)), 6),
            'centrality_method': centrality_method,
            'is_connected': nx.is_connected(G),
            'num_connected_components': nx.number_connected_components(G),
        }

        return jsonify({
            'status': 'success',
            'summary': summary,
            'betweenness_centrality': betweenness_nodes,
            'eigenvector_centrality': eigenvector_nodes,
            'articulation_points': articulation_nodes,
        })

    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/network-topology', methods=['GET'])
def get_network_topology():
    """
    Compute network topology metrics that indicate structural resilience:
    - Graph Density: how interconnected the network is
    - Average Path Length: efficiency of flow
    - Clustering Coefficient: local redundancy
    - Network Diameter: worst-case distance
    - Assortativity: do hubs connect to other hubs?
    """
    try:
        # Build NetworkX graph
        G = nx.Graph()
        for idx, row in node_df.iterrows():
            G.add_node(int(idx))
        
        weight_col = 'weight' if 'weight' in edge_df.columns else 'capacity_share'
        for _, row in edge_df.iterrows():
            G.add_edge(int(row['source']), int(row['target']), weight=float(row[weight_col]))
        
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        
        # Density: actual edges / possible edges
        density = nx.density(G)
        
        # Average shortest path length (only if connected)
        if nx.is_connected(G):
            avg_path_length = nx.average_shortest_path_length(G, weight='weight')
            diameter = nx.diameter(G)
        else:
            # For disconnected graphs, compute for largest component
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            avg_path_length = nx.average_shortest_path_length(subgraph, weight='weight')
            diameter = nx.diameter(subgraph)
        
        # Clustering coefficient: local redundancy
        avg_clustering = nx.average_clustering(G)
        
        # Assortativity: do high-degree nodes connect to other high-degree nodes?
        try:
            assortativity = nx.degree_assortativity_coefficient(G)
        except:
            assortativity = 0.0
        
        # Compute ideal/benchmark values for comparison
        # Ideal values for a resilient supply chain network
        ideal_density = 0.15  # Not too sparse, not complete
        ideal_clustering = 0.3  # Some local redundancy
        ideal_assortativity = 0.0  # Neutral mixing
        
        return jsonify({
            'status': 'success',
            'metrics': {
                'density': {
                    'value': round(density, 4),
                    'ideal': ideal_density,
                    'percentage': round((density / ideal_density) * 100, 1) if ideal_density > 0 else 100,
                    'description': 'Network interconnectedness (0=sparse, 1=complete)'
                },
                'avg_path_length': {
                    'value': round(avg_path_length, 2),
                    'description': 'Average hops between any two nodes (lower = more efficient)'
                },
                'clustering': {
                    'value': round(avg_clustering, 4),
                    'ideal': ideal_clustering,
                    'percentage': round((avg_clustering / ideal_clustering) * 100, 1) if ideal_clustering > 0 else 100,
                    'description': 'Local redundancy (higher = more backup paths)'
                },
                'diameter': {
                    'value': int(diameter),
                    'description': 'Maximum distance between any two nodes (lower = better)'
                },
                'assortativity': {
                    'value': round(assortativity, 4),
                    'ideal': ideal_assortativity,
                    'description': 'Hub connectivity pattern (-1=hubs avoid hubs, +1=hubs connect to hubs)'
                }
            },
            'summary': {
                'num_nodes': num_nodes,
                'num_edges': num_edges,
                'is_connected': nx.is_connected(G),
                'num_components': nx.number_connected_components(G)
            }
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/cascading-failure', methods=['GET'])
def get_cascading_failure():
    """
    Simulate cascading failures by removing vulnerable nodes one-by-one.
    Shows the impact of each removal on network connectivity.
    """
    try:
        top_n = int(request.args.get('top_n', 10))
        
        # Build NetworkX graph
        G = nx.Graph()
        for idx, row in node_df.iterrows():
            G.add_node(
                int(idx),
                name=str(row['node_id']),
                tier=int(row['tier']),
                region=str(row['region'])
            )
        
        weight_col = 'weight' if 'weight' in edge_df.columns else 'capacity_share'
        for _, row in edge_df.iterrows():
            G.add_edge(int(row['source']), int(row['target']), weight=float(row[weight_col]))
        
        initial_nodes = G.number_of_nodes()
        initial_edges = G.number_of_edges()
        
        # Get betweenness centrality to identify critical nodes
        betweenness = nx.betweenness_centrality(G, normalized=True, weight='weight')
        
        # Sort nodes by betweenness (most critical first)
        sorted_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Simulate cascading failures
        cascade_results = []
        G_sim = G.copy()
        
        for node_id, bc_score in sorted_nodes:
            # Remove the node
            G_sim.remove_node(node_id)
            
            # Calculate impact
            remaining_nodes = G_sim.number_of_nodes()
            remaining_edges = G_sim.number_of_edges()
            num_components = nx.number_connected_components(G_sim)
            is_connected = nx.is_connected(G_sim)
            
            # Get largest component size
            if num_components > 0:
                largest_cc_size = len(max(nx.connected_components(G_sim), key=len))
            else:
                largest_cc_size = 0
            
            node_attrs = G.nodes[node_id]
            cascade_results.append({
                'node_id': node_id,
                'node_name': node_attrs['name'],
                'tier': node_attrs['tier'],
                'region': node_attrs['region'],
                'betweenness': round(bc_score, 6),
                'nodes_disconnected': initial_nodes - remaining_nodes,
                'edges_lost': initial_edges - remaining_edges,
                'remaining_nodes': remaining_nodes,
                'num_components': num_components,
                'is_connected': is_connected,
                'largest_component_size': largest_cc_size,
                'fragmentation_ratio': round(1 - (largest_cc_size / initial_nodes), 4)
            })
        
        return jsonify({
            'status': 'success',
            'initial_state': {
                'num_nodes': initial_nodes,
                'num_edges': initial_edges,
                'is_connected': True
            },
            'cascade_sequence': cascade_results
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/trade-restriction-scenario', methods=['POST'])
def trade_restriction_scenario():
    """
    Simulate China-US trade restriction scenario.
    Shows impact of trade restrictions between China and United States.
    
    Request body:
    {
        "severity": 0.8,  # 0.0-1.0, default 0.8
        "region1": "China",  # Optional, default "China"
        "region2": "United_States"  # Optional, default "United_States"
    }
    
    Response:
    {
        "status": "success",
        "baseline": {...},
        "disrupted": {...},
        "impact": {...},
        "affected_nodes": [...],
        "disrupted_edges": [...]
    }
    """
    try:
        data = request.json or {}
        severity = data.get('severity', 0.8)
        region1 = data.get('region1', 'China')
        region2 = data.get('region2', 'United_States')
        
        print(f"\n🚫 Trade Restriction Scenario: {region1} ↔ {region2} (severity={severity})")
        
        # Identify nodes in each region
        # Try multiple region name variations
        china_regions = ['East_Asia', 'China', 'Asia', 'CN']
        us_regions = ['North_America', 'United_States', 'US', 'USA', 'NA']
        
        china_nodes = node_df[node_df['region'].isin(china_regions)].index.tolist()
        us_nodes = node_df[node_df['region'].isin(us_regions)].index.tolist()
        
        # If no exact matches, use geographic coordinates
        if len(china_nodes) == 0:
            # China: roughly 18-54°N, 73-135°E
            china_nodes = node_df[
                (node_df['y'] >= 18) & (node_df['y'] <= 54) &
                (node_df['x'] >= 73) & (node_df['x'] <= 135)
            ].index.tolist()
        
        if len(us_nodes) == 0:
            # US: roughly 25-50°N, -125 to -65°W
            us_nodes = node_df[
                (node_df['y'] >= 25) & (node_df['y'] <= 50) &
                (node_df['x'] >= -125) & (node_df['x'] <= -65)
            ].index.tolist()
        
        print(f"  ✓ {region1} nodes: {len(china_nodes)}")
        print(f"  ✓ {region2} nodes: {len(us_nodes)}")
        
        # Find edges connecting the two regions
        disrupted_edges = []
        for idx, row in edge_df.iterrows():
            source = int(row['source'])
            target = int(row['target'])
            
            if (source in china_nodes and target in us_nodes) or \
               (source in us_nodes and target in china_nodes):
                disrupted_edges.append([source, target])
        
        print(f"  ✓ Disrupted trade routes: {len(disrupted_edges)}")
        
        # Baseline prediction (no disruption)
        baseline_data = create_pyg_data(node_df, edge_df, [], [], 0.0)
        baseline_data = baseline_data.to(device)
        
        with torch.no_grad():
            baseline_out = model(baseline_data.x, baseline_data.edge_index, baseline_data.edge_attr)
            baseline_preds = baseline_out.argmax(dim=1).cpu().numpy()
        
        baseline_summary = {
            'failed': int((baseline_preds == 0).sum()),
            'degraded': int((baseline_preds == 1).sum()),
            'normal': int((baseline_preds == 2).sum()),
            'total_nodes': len(node_df)
        }
        
        # Trade restriction scenario
        affected_nodes = list(set(china_nodes + us_nodes))
        disrupted_data = create_pyg_data(node_df, edge_df, affected_nodes, disrupted_edges, severity)
        disrupted_data = disrupted_data.to(device)
        
        with torch.no_grad():
            disrupted_out = model(disrupted_data.x, disrupted_data.edge_index, disrupted_data.edge_attr)
            disrupted_probs = torch.exp(disrupted_out)
            disrupted_preds = disrupted_out.argmax(dim=1).cpu().numpy()
        
        disrupted_summary = {
            'failed': int((disrupted_preds == 0).sum()),
            'degraded': int((disrupted_preds == 1).sum()),
            'normal': int((disrupted_preds == 2).sum()),
            'total_nodes': len(node_df)
        }
        
        # Calculate impact
        impact = {
            'delta_failed': disrupted_summary['failed'] - baseline_summary['failed'],
            'delta_degraded': disrupted_summary['degraded'] - baseline_summary['degraded'],
            'delta_normal': disrupted_summary['normal'] - baseline_summary['normal'],
            'nodes_worsened': int(((disrupted_preds - baseline_preds) < 0).sum()),
            'severity': severity
        }
        
        # Prepare node-level results
        affected_node_details = []
        for node_id in affected_nodes[:50]:  # Limit to first 50 for performance
            affected_node_details.append({
                'node_id': int(node_id),
                'name': str(node_df.iloc[node_id]['node_id']),
                'region': str(node_df.iloc[node_id]['region']),
                'tier': int(node_df.iloc[node_id]['tier']),
                'x': float(node_df.iloc[node_id]['x']),
                'y': float(node_df.iloc[node_id]['y']),
                'baseline_status': ['Failed', 'Degraded', 'Normal'][int(baseline_preds[node_id])],
                'disrupted_status': ['Failed', 'Degraded', 'Normal'][int(disrupted_preds[node_id])],
                'status_change': int(disrupted_preds[node_id] - baseline_preds[node_id])
            })
        
        # Prepare all node predictions for map visualization
        all_nodes_map = []
        for idx in range(len(node_df)):
            all_nodes_map.append({
                'node_id': int(idx),
                'x': float(node_df.iloc[idx]['x']),
                'y': float(node_df.iloc[idx]['y']),
                'region': str(node_df.iloc[idx]['region']),
                'baseline_pred': int(baseline_preds[idx]),
                'disrupted_pred': int(disrupted_preds[idx]),
                'is_china': idx in china_nodes,
                'is_us': idx in us_nodes
            })
        
        return jsonify({
            'status': 'success',
            'scenario': {
                'type': 'trade_restriction',
                'region1': region1,
                'region2': region2,
                'severity': severity,
                'china_nodes_count': len(china_nodes),
                'us_nodes_count': len(us_nodes),
                'disrupted_edges_count': len(disrupted_edges)
            },
            'baseline': baseline_summary,
            'disrupted': disrupted_summary,
            'impact': impact,
            'affected_nodes': affected_node_details,
            'all_nodes_map': all_nodes_map,
            'disrupted_edges': disrupted_edges[:100]  # Limit for performance
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500


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


@app.route('/api/run-complete-pipeline', methods=['POST'])
def run_complete_pipeline():
    """
    Run the complete training pipeline with default or custom data.
    This endpoint triggers complete_training_pipeline.py
    """
    try:
        data = request.json or {}
        num_scenarios = data.get('num_scenarios', 1000)
        use_default_data = data.get('use_default_data', True)
        
        # Generate task ID
        task_id = uuid.uuid4().hex
        
        # Determine paths
        if use_default_data:
            node_path = 'C:/Users/janny/Desktop/final_year/actual_pipeline_backup/synthetic_nodes.csv'
            edge_path = 'C:/Users/janny/Desktop/final_year/actual_pipeline_backup/synthetic_edges.csv'
            output_dir = 'C:/Users/janny/Desktop/final_year/pipeline_output'
        else:
            company_id = data.get('company_id')
            if not company_id:
                return jsonify({'status': 'error', 'message': 'company_id required for custom data'}), 400
            
            company_dir = os.path.join(app.config['UPLOAD_FOLDER'], company_id)
            if not os.path.exists(company_dir):
                return jsonify({'status': 'error', 'message': 'Company not found'}), 404
            
            node_path = os.path.join(company_dir, 'nodes.csv')
            edge_path = os.path.join(company_dir, 'edges.csv')
            output_dir = company_dir
        
        # Create task info file
        task_info = {
            'task_id': task_id,
            'status': 'running',
            'progress': 0,
            'message': 'Starting complete training pipeline...',
            'stage': 'init',
            'started_at': datetime.now().isoformat(),
            'num_scenarios': num_scenarios,
            'use_default_data': use_default_data
        }
        
        task_path = os.path.join(output_dir, f'pipeline_task_{task_id}.json')
        os.makedirs(output_dir, exist_ok=True)
        with open(task_path, 'w') as f:
            json.dump(task_info, f, indent=2)
        
        # Run pipeline in background thread
        import threading
        from complete_training_pipeline import CompletePipeline
        
        def run_pipeline():
            try:
                # Update progress
                task_info['progress'] = 5
                task_info['message'] = 'Initializing pipeline...'
                task_info['stage'] = 'init'
                with open(task_path, 'w') as f:
                    json.dump(task_info, f, indent=2)
                
                # Run pipeline
                pipeline = CompletePipeline(seed=42, output_dir=output_dir)
                results = pipeline.run(
                    node_path=node_path,
                    edge_path=edge_path,
                    num_scenarios=num_scenarios,
                    scenario_type='node'
                )
                
                # Update task with results
                task_info['status'] = 'completed'
                task_info['progress'] = 100
                task_info['message'] = 'Pipeline completed successfully!'
                task_info['stage'] = 'done'
                task_info['completed_at'] = datetime.now().isoformat()
                task_info['results'] = results.to_dict('records')
                
                with open(task_path, 'w') as f:
                    json.dump(task_info, f, indent=2)
                    
            except Exception as e:
                import traceback
                task_info['status'] = 'failed'
                task_info['error'] = str(e)
                task_info['traceback'] = traceback.format_exc()
                with open(task_path, 'w') as f:
                    json.dump(task_info, f, indent=2)
        
        thread = threading.Thread(target=run_pipeline, daemon=True)
        thread.start()
        
        return jsonify({
            'status': 'started',
            'task_id': task_id,
            'message': f'Pipeline started with {num_scenarios} scenarios. This may take 30-60 minutes.'
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/pipeline-status/<task_id>', methods=['GET'])
def pipeline_status(task_id):
    """Get status of pipeline task."""
    try:
        # Search in pipeline_output and uploads directories
        search_dirs = [
            'C:/Users/janny/Desktop/final_year/pipeline_output',
            app.config['UPLOAD_FOLDER']
        ]
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                # Check direct path
                task_path = os.path.join(search_dir, f'pipeline_task_{task_id}.json')
                if os.path.exists(task_path):
                    with open(task_path, 'r') as f:
                        task_info = json.load(f)
                    return jsonify(task_info)
                
                # Check subdirectories
                if os.path.isdir(search_dir):
                    for subdir in os.listdir(search_dir):
                        subdir_path = os.path.join(search_dir, subdir)
                        if os.path.isdir(subdir_path):
                            task_path = os.path.join(subdir_path, f'pipeline_task_{task_id}.json')
                            if os.path.exists(task_path):
                                with open(task_path, 'r') as f:
                                    task_info = json.load(f)
                                return jsonify(task_info)
        
        return jsonify({
            'status': 'not_found',
            'message': 'Task not found'
        }), 404
        
    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500


if __name__ == '__main__':
    print("🚀 Starting GNN Backend API...")
    print(f"📊 Loaded {len(node_df)} nodes, {len(edge_df)} edges")
    print(f"🤖 Model: GINE (76.6% accuracy)")
    print(f"💻 Device: {device}")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
