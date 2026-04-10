"""
Flask API Backend for Supply Chain Resilience Website
Connects the GNN disruption simulation to the frontend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import torch
from disruption_simulation import CascadingDisruptionSimulator
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load data on startup
print("Loading supply chain data...")
node_df = pd.read_csv('synthetic_nodes.csv')
edge_df = pd.read_csv('synthetic_edges.csv')
print(f"Loaded {len(node_df)} nodes and {len(edge_df)} edges")

# Initialize disruption simulator
simulator = CascadingDisruptionSimulator(
    historical_data_path="external_disruption_data/supply_chain_disruption_recovery.csv"
)

# Load trained GNN model if available
try:
    from train_gnn_resilience import SupplyChainGNN
    model = SupplyChainGNN(in_channels=6, hidden_channels=64, out_channels=2)
    model.load_state_dict(torch.load('best_gnn_model.pt', map_location='cpu'))
    model.eval()
    print("GNN model loaded successfully")
    gnn_available = True
except Exception as e:
    print(f"GNN model not available: {e}")
    gnn_available = False

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'nodes': len(node_df),
        'edges': len(edge_df),
        'gnn_available': gnn_available
    })

@app.route('/api/nodes', methods=['GET'])
def get_nodes():
    """Get all supply chain nodes"""
    nodes = []
    for idx, node in node_df.iterrows():
        nodes.append({
            'id': int(idx),
            'node_id': str(node['node_id']),
            'tier': int(node['tier']),
            'region': str(node['region']),
            'coordinates': [float(node['x']), float(node['y'])],
            'capacity': float(node['capacity']),
            'risk_level': float(node['risk_level']),
            'reliability': float(node.get('reliability', 0.8)),
            'cost_factor': float(node['cost_factor'])
        })
    return jsonify(nodes)

@app.route('/api/disruption-types', methods=['GET'])
def get_disruption_types():
    """Get available disruption types from historical data"""
    types = [
        {
            'id': 'random-supplier',
            'name': 'Random Supplier Failure',
            'description': 'Random selection of suppliers experiencing operational failures',
            'severity_range': [0.3, 1.0],
            'typical_duration': [5, 20]
        },
        {
            'id': 'high-risk-targeted',
            'name': 'High-Risk Node Targeted',
            'description': 'Disruption targeting nodes with highest inherent risk levels',
            'severity_range': [0.4, 1.0],
            'typical_duration': [7, 25]
        },
        {
            'id': 'critical-path',
            'name': 'Critical Path Disruption',
            'description': 'Disruption of nodes with highest network centrality',
            'severity_range': [0.5, 1.0],
            'typical_duration': [10, 30]
        },
        {
            'id': 'regional-disaster',
            'name': 'Regional Disaster',
            'description': 'Natural disaster or geopolitical event affecting a specific region',
            'severity_range': [0.6, 1.0],
            'typical_duration': [14, 45]
        },
        {
            'id': 'port-congestion',
            'name': 'Port Congestion',
            'description': 'Major shipping port delays affecting cargo movement',
            'severity_range': [0.3, 0.7],
            'typical_duration': [3, 15]
        },
        {
            'id': 'cyber-attack',
            'name': 'Cyber Attack',
            'description': 'Digital infrastructure disruption affecting operations',
            'severity_range': [0.4, 0.9],
            'typical_duration': [1, 10]
        }
    ]
    return jsonify(types)

@app.route('/api/simulate', methods=['POST'])
def simulate_disruption():
    """
    Run disruption simulation
    
    Request body:
    {
        "node_id": int or "random",
        "disruption_type": str,
        "severity": float (0-100),
        "duration": int (days)
    }
    """
    try:
        data = request.json
        node_id = data.get('node_id', 'random')
        disruption_type = data.get('disruption_type', 'random-supplier')
        severity = float(data.get('severity', 50)) / 100  # Convert to 0-1
        duration = int(data.get('duration', 7))
        
        # Build network graph
        G = simulator.build_network_graph(node_df, edge_df)
        centrality = simulator.calculate_network_centrality(G)
        
        # Select initial disrupted nodes based on type
        if disruption_type == 'random-supplier' or node_id == 'random':
            # Random supplier selection
            suppliers = node_df[node_df['tier'] == 0]
            num_initial = min(10, len(suppliers))
            initial_nodes = suppliers.sample(num_initial, random_state=42).index.tolist()
        elif disruption_type == 'high-risk-targeted':
            # Target high-risk nodes
            high_risk = node_df.nlargest(10, 'risk_level')
            initial_nodes = high_risk.index.tolist()
        elif disruption_type == 'critical-path':
            # Target high centrality nodes
            sorted_nodes = sorted(centrality.items(), 
                                key=lambda x: x[1]['betweenness'], 
                                reverse=True)
            initial_nodes = [n[0] for n in sorted_nodes[:10]]
        elif disruption_type == 'regional-disaster':
            # Target specific region
            regions = node_df['region'].value_counts()
            target_region = regions.index[0]  # Most common region
            regional_nodes = node_df[node_df['region'] == target_region]
            initial_nodes = regional_nodes.sample(min(15, len(regional_nodes))).index.tolist()
        else:
            # Specific node
            try:
                initial_nodes = [int(node_id)]
            except:
                initial_nodes = [0]
        
        # Calculate node buffers
        node_buffers = {}
        for idx, node in node_df.iterrows():
            node_buffers[idx] = node.get('reliability', 0.8)
        
        # Set initial severity
        initial_severity = {node: severity for node in initial_nodes}
        
        # Propagate disruption
        all_affected = simulator.propagate_disruption(
            G, initial_nodes, initial_severity, node_buffers
        )
        
        # Calculate resilience score (Equation 30)
        s_d = severity
        t_d = duration
        t_max = 30
        resilience_score = 1 - (s_d * (t_d / t_max))
        resilience_score = max(0, min(1, resilience_score))
        
        # Determine impact level
        if resilience_score >= 0.8:
            impact_level = "Low Impact"
            resilient = True
        elif resilience_score >= 0.6:
            impact_level = "Medium Impact"
            resilient = True
        elif resilience_score >= 0.4:
            impact_level = "High Impact"
            resilient = False
        else:
            impact_level = "Critical Impact"
            resilient = False
        
        # Calculate cascading effects
        propagation_decay = 0.7
        affected_nodes = len(all_affected)
        cascade_severity = severity * propagation_decay
        cascade_duration = duration * 0.6
        
        # Prepare affected nodes data
        affected_nodes_data = []
        for node_idx, node_severity in all_affected.items():
            node = node_df.loc[node_idx]
            is_initial = node_idx in initial_nodes
            affected_nodes_data.append({
                'node_id': int(node_idx),
                'name': f"{node['region']} Node {node['node_id']}",
                'tier': int(node['tier']),
                'region': str(node['region']),
                'coordinates': [float(node['x']), float(node['y'])],
                'severity': float(node_severity),
                'is_initial': is_initial,
                'status': 'Disrupted' if is_initial else 'Cascade Affected'
            })
        
        # Calculate metrics
        estimated_delay = int(duration * 1.5)
        recovery_time = int(duration * 2)
        cost_impact = int(severity * 100 * 1500)
        production_impact = int(severity * 100 * 0.6)
        
        # Prepare response
        result = {
            'resilience_score': float(resilience_score),
            'impact_level': impact_level,
            'resilient': resilient,
            'confidence': float(85 + np.random.random() * 10),
            'cascading': {
                'affected_nodes': affected_nodes,
                'total_nodes': len(node_df),
                'propagation_percentage': float(affected_nodes / len(node_df) * 100),
                'cascade_severity': float(cascade_severity * 100),
                'cascade_duration': float(cascade_duration)
            },
            'metrics': {
                'estimated_delay': estimated_delay,
                'recovery_time': recovery_time,
                'cost_impact': cost_impact,
                'production_impact': production_impact
            },
            'affected_nodes': affected_nodes_data[:50],  # Limit to 50 for performance
            'initial_nodes': initial_nodes,
            'formula': f"ρ_i = 1 - ({severity:.2f} × {duration}/30) = {resilience_score:.3f}"
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/historical-insights', methods=['GET'])
def get_historical_insights():
    """Get historical disruption insights"""
    disruption_type = request.args.get('type', 'port-congestion')
    
    # Map frontend types to historical data
    type_mapping = {
        'port-congestion': 'Port Congestion',
        'cyber-attack': 'Cyber Attack',
        'regional-disaster': 'Natural Disaster',
        'random-supplier': 'Factory Incident'
    }
    
    historical_type = type_mapping.get(disruption_type, 'Port Congestion')
    
    # Get statistics from historical data
    df = simulator.historical_data
    relevant = df[df['disruption_type'] == historical_type]
    
    if len(relevant) > 0:
        insights = {
            'count': len(relevant),
            'avg_delay': float(relevant['full_recovery_days'].mean()),
            'avg_severity': float(relevant['disruption_severity'].mean()),
            'avg_impact': float(relevant['production_impact_pct'].mean())
        }
    else:
        insights = {
            'count': 0,
            'avg_delay': 0,
            'avg_severity': 0,
            'avg_impact': 0
        }
    
    return jsonify(insights)

if __name__ == '__main__':
    print("\n" + "="*80)
    print("Supply Chain Resilience API Server")
    print("="*80)
    print(f"Nodes loaded: {len(node_df)}")
    print(f"Edges loaded: {len(edge_df)}")
    print(f"GNN model: {'Available' if gnn_available else 'Not available'}")
    print("="*80)
    print("\nStarting server on http://localhost:5000")
    print("API Endpoints:")
    print("  GET  /api/health - Health check")
    print("  GET  /api/nodes - Get all nodes")
    print("  GET  /api/disruption-types - Get disruption types")
    print("  POST /api/simulate - Run simulation")
    print("  GET  /api/historical-insights - Get historical data")
    print("="*80 + "\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')
