"""
Realistic Stochastic Simulation Engine - Production Impact Based
Generates scenarios using percentage-based production impact with selective feature exposure.

Key Changes:
1. ALL nodes know their own buffer (realistic - companies track inventory)
2. ONLY disrupted node knows production_impact_pct (others must infer from cascade)
3. Use percentage-based impact instead of absolute shock
4. Add is_disrupted flag to identify epicenter
5. 7 features per node: [capacity, cost_factor, risk_level, reliability, buffer, production_impact_pct, is_disrupted]

Core Logic:
- TTS (Time To Survive) = Buffer capacity (ALL nodes know this)
- TTR (Time To Recover) = Production impact % (only disrupted node knows)
- Label 1 (Resilient): Buffer >= (Capacity * Production Impact %)
- Label 0 (Failed): Buffer < (Capacity * Production Impact %)
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import networkx as nx
from collections import deque
import os
import json
from tqdm import tqdm

class RealisticDisruptionSimulator:
    """
    Simulates supply chain disruptions with realistic feature exposure.
    """
    
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
        
    def load_graph(self, node_df, edge_df):
        """Load supply chain graph."""
        print("\n📊 Loading supply chain graph...")
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for idx, row in node_df.iterrows():
            G.add_node(
                idx,
                capacity=row['capacity'],
                risk_level=row['risk_level'],
                tier=row['tier'],
                region=row['region'],
                cost_factor=row['cost_factor'],
                reliability=row['reliability']
            )
        
        # Add edges with weights (use capacity_share as weight)
        for idx, row in edge_df.iterrows():
            G.add_edge(
                row['source'],
                row['target'],
                weight=row.get('weight', row.get('capacity_share', 1.0))
            )
        
        print(f"  ✓ Loaded {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def calculate_base_buffers(self, G):
        """
        Calculate base buffer for each node (30-70% of capacity).
        This represents inventory/safety stock that ALL nodes know.
        """
        buffers = {}
        for node in G.nodes():
            base_capacity = G.nodes[node]['capacity']
            # Base buffer: 30-70% of capacity
            buffers[node] = base_capacity * np.random.uniform(0.3, 0.7)
        return buffers
    
    def select_initial_node(self, G, scenario_type='random'):
        """Select initial disrupted node based on scenario type."""
        nodes = list(G.nodes())
        
        if scenario_type == 'central':
            # High betweenness centrality nodes
            centrality = nx.betweenness_centrality(G)
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            top_20_pct = max(1, len(sorted_nodes) // 5)
            return np.random.choice([n for n, _ in sorted_nodes[:top_20_pct]])
        
        elif scenario_type == 'high_risk':
            # Nodes with high risk_level
            risk_nodes = [(n, G.nodes[n]['risk_level']) for n in nodes]
            sorted_nodes = sorted(risk_nodes, key=lambda x: x[1], reverse=True)
            top_20_pct = max(1, len(sorted_nodes) // 5)
            return np.random.choice([n for n, _ in sorted_nodes[:top_20_pct]])
        
        elif scenario_type == 'high_capacity':
            # High capacity nodes (major hubs)
            capacity_nodes = [(n, G.nodes[n]['capacity']) for n in nodes]
            sorted_nodes = sorted(capacity_nodes, key=lambda x: x[1], reverse=True)
            top_20_pct = max(1, len(sorted_nodes) // 5)
            return np.random.choice([n for n, _ in sorted_nodes[:top_20_pct]])
        
        else:  # random
            return np.random.choice(nodes)
    
    def simulate_cascade(self, G, base_buffers, initial_node, initial_impact_pct):
        """
        Simulate cascading disruption with production impact percentage.
        
        Args:
            G: NetworkX graph
            base_buffers: Dict of base buffer values for each node
            initial_node: Node where disruption starts
            initial_impact_pct: Initial production impact as percentage (0.0 to 1.0)
        
        Returns:
            Dict with results for each affected node
        """
        # Add stochastic variance to buffers (±20%)
        buffers = {
            node: base_buffers[node] * np.random.uniform(0.8, 1.2)
            for node in G.nodes()
        }
        
        # Track results for each node
        results = {}
        visited = set()
        
        # BFS queue: (node_id, production_impact_pct)
        queue = deque([(initial_node, initial_impact_pct)])
        
        while queue:
            current_node, impact_pct = queue.popleft()
            
            if current_node in visited:
                continue
            
            visited.add(current_node)
            
            # Get node properties
            capacity = G.nodes[current_node]['capacity']
            buffer = buffers[current_node]
            
            # Calculate impact in absolute units
            impact_units = capacity * impact_pct
            
            # Determine if node survives
            remaining_impact = max(0, impact_units - buffer)
            
            # Label: 1 if buffer absorbed all impact, 0 if failed
            label = 1 if remaining_impact == 0 else 0
            
            # Store results
            results[current_node] = {
                'buffer': buffer,
                'production_impact_pct': impact_pct,
                'production_impact_units': impact_units,
                'remaining_impact': remaining_impact,
                'label': label,
                'is_disrupted': 1 if current_node == initial_node else 0
            }
            
            # If node failed, propagate impact downstream
            if remaining_impact > 0:
                downstream = list(G.successors(current_node))
                
                if downstream:
                    # Calculate total outgoing flow
                    total_flow = sum(G[current_node][neighbor].get('weight', 1.0) 
                                   for neighbor in downstream)
                    
                    # Distribute remaining impact to downstream nodes
                    for neighbor in downstream:
                        edge_weight = G[current_node][neighbor].get('weight', 1.0)
                        proportion = edge_weight / total_flow
                        
                        # Base impact to this neighbor
                        base_impact_units = remaining_impact * proportion
                        
                        # Apply stochastic decay factor (0.5 to 1.2)
                        # Models real-world uncertainty (backup suppliers vs panic ordering)
                        decay_factor = np.random.uniform(0.5, 1.2)
                        actual_impact_units = base_impact_units * decay_factor
                        
                        # Convert back to percentage for neighbor
                        neighbor_capacity = G.nodes[neighbor]['capacity']
                        neighbor_impact_pct = actual_impact_units / neighbor_capacity
                        
                        # Add to queue
                        queue.append((neighbor, neighbor_impact_pct))
        
        return results
    
    def generate_scenarios(self, G, base_buffers, num_scenarios=10000):
        """Generate multiple disruption scenarios."""
        print(f"\n🎲 Generating {num_scenarios} disruption scenarios...")
        
        scenarios = []
        scenario_types = ['random', 'central', 'high_risk', 'high_capacity']
        
        for i in tqdm(range(num_scenarios), desc="Simulating scenarios"):
            # Select scenario type
            scenario_type = scenario_types[i % len(scenario_types)]
            
            # Select initial node
            initial_node = self.select_initial_node(G, scenario_type)
            
            # Random initial production impact (10% to 100%)
            initial_impact_pct = np.random.uniform(0.1, 1.0)
            
            # Simulate cascade
            results = self.simulate_cascade(G, base_buffers, initial_node, initial_impact_pct)
            
            scenarios.append({
                'scenario_id': i,
                'scenario_type': scenario_type,
                'initial_node': initial_node,
                'initial_impact_pct': initial_impact_pct,
                'results': results
            })
        
        # Calculate statistics
        total_affected = sum(len(s['results']) for s in scenarios)
        avg_affected = total_affected / num_scenarios
        
        print(f"\n  ✓ Generated {num_scenarios} scenarios")
        print(f"  ✓ Average nodes affected per scenario: {avg_affected:.1f}")
        
        return scenarios
    
    def create_pyg_data_objects(self, G, node_df, edge_df, scenarios):
        """
        Convert scenarios to PyTorch Geometric Data objects.
        
        Feature design (HARDER VERSION):
        - ALL nodes get buffer (realistic - companies know their inventory)
        - NO production_impact_pct (hidden from everyone)
        - NO is_disrupted flag (must infer from graph)
        
        Features (5 total):
        [0] capacity
        [1] cost_factor
        [2] risk_level
        [3] reliability
        [4] buffer (ALL nodes)
        
        GNN must use graph structure to infer:
        - Which node is disrupted
        - Impact magnitude
        - Cascade patterns
        """
        print("\n💾 Creating PyG Data objects with HARDER feature design...")
        
        # Prepare base features (first 4 features)
        base_features = torch.tensor(
            node_df[['capacity', 'cost_factor', 'risk_level', 'reliability']].values,
            dtype=torch.float
        )
        
        # Create edge_index
        edge_index = torch.tensor([
            edge_df['source'].values,
            edge_df['target'].values
        ], dtype=torch.long)
        
        num_nodes = len(node_df)
        data_objects = []
        
        for scenario in tqdm(scenarios, desc="Creating Data objects"):
            # Initialize features: [base_features + buffer] (5 features total)
            x = torch.zeros((num_nodes, base_features.shape[1] + 1), dtype=torch.float)
            x[:, :base_features.shape[1]] = base_features
            
            # Initialize labels and mask
            y = torch.full((num_nodes,), -1, dtype=torch.long)
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            
            # Fill in scenario-specific data
            for node_id, result in scenario['results'].items():
                node_id = int(node_id)
                
                # Feature 4: buffer (ALL nodes know this)
                x[node_id, 4] = result['buffer']
                
                # NO production_impact_pct - hidden from everyone
                # NO is_disrupted flag - must infer from graph
                
                # Set label
                y[node_id] = result['label']
                train_mask[node_id] = True
            
            # Create Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                y=y,
                train_mask=train_mask
            )
            
            # Add metadata
            data.scenario_id = scenario['scenario_id']
            data.scenario_type = scenario['scenario_type']
            data.initial_node = scenario['initial_node']
            data.initial_impact_pct = scenario['initial_impact_pct']
            
            data_objects.append(data)
        
        print(f"  ✓ Created {len(data_objects)} Data objects")
        print(f"  ✓ Feature dimensions: {data_objects[0].x.shape}")
        print(f"  ✓ Features: [capacity, cost_factor, risk_level, reliability, buffer]")
        
        return data_objects
    
    def save_scenarios(self, data_objects, output_dir='scenario_graphs_realistic'):
        """Save scenarios to disk."""
        print(f"\n💾 Saving scenarios to {output_dir}/...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each scenario
        for data in tqdm(data_objects, desc="Saving scenarios"):
            filename = f"scenario_{data.scenario_id:05d}.pt"
            filepath = os.path.join(output_dir, filename)
            torch.save(data, filepath)
        
        # Save metadata
        metadata = {
            'num_scenarios': len(data_objects),
            'num_nodes': data_objects[0].x.shape[0],
            'num_features': data_objects[0].x.shape[1],
            'feature_names': [
                'capacity',
                'cost_factor',
                'risk_level',
                'reliability',
                'buffer'
            ],
            'feature_description': {
                'capacity': 'Node production capacity (all nodes)',
                'cost_factor': 'Operational cost factor (all nodes)',
                'risk_level': 'Baseline vulnerability (all nodes)',
                'reliability': 'Historical performance (all nodes)',
                'buffer': 'Available inventory - ALL nodes know this (realistic)'
            },
            'label_logic': 'Label 1 (Resilient) if buffer >= (capacity * production_impact_pct), else 0 (Failed). NOTE: production_impact_pct is HIDDEN - not in features!'
        }
        
        metadata_path = os.path.join(output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Saved {len(data_objects)} scenarios")
        print(f"  ✓ Saved metadata to {metadata_path}")
        
        return output_dir


def main():
    """Main execution pipeline."""
    print("="*70)
    print("REALISTIC STOCHASTIC DISRUPTION SIMULATION")
    print("="*70)
    print("\nKey Features (HARDER VERSION):")
    print("  ✓ ALL nodes know their own buffer (realistic)")
    print("  ✓ NO production_impact_pct in features (hidden from everyone)")
    print("  ✓ NO is_disrupted flag (must infer from graph)")
    print("  ✓ GNN must use graph structure to infer hidden information")
    print("  ✓ 5 features per node: [capacity, cost, risk, reliability, buffer]")
    
    # Load data
    print("\n" + "="*70)
    print("STEP 1: LOADING DATA")
    print("="*70)
    
    node_df = pd.read_csv('synthetic_nodes.csv')
    edge_df = pd.read_csv('synthetic_edges.csv')
    
    print(f"  ✓ Loaded {len(node_df)} nodes")
    print(f"  ✓ Loaded {len(edge_df)} edges")
    
    # Initialize simulator
    simulator = RealisticDisruptionSimulator(seed=42)
    
    # Load graph
    print("\n" + "="*70)
    print("STEP 2: BUILDING GRAPH")
    print("="*70)
    
    G = simulator.load_graph(node_df, edge_df)
    
    # Calculate base buffers
    print("\n" + "="*70)
    print("STEP 3: CALCULATING BASE BUFFERS")
    print("="*70)
    
    base_buffers = simulator.calculate_base_buffers(G)
    print(f"  ✓ Calculated buffers for {len(base_buffers)} nodes")
    print(f"  ✓ Buffer range: {min(base_buffers.values()):.1f} - {max(base_buffers.values()):.1f}")
    
    # Generate scenarios
    print("\n" + "="*70)
    print("STEP 4: GENERATING SCENARIOS")
    print("="*70)
    
    num_scenarios = 10000
    scenarios = simulator.generate_scenarios(G, base_buffers, num_scenarios=num_scenarios)
    
    # Analyze scenario distribution
    print("\n📊 Scenario Statistics:")
    resilient_count = sum(
        sum(1 for r in s['results'].values() if r['label'] == 1)
        for s in scenarios
    )
    failed_count = sum(
        sum(1 for r in s['results'].values() if r['label'] == 0)
        for s in scenarios
    )
    total_labels = resilient_count + failed_count
    
    print(f"  Total labeled nodes: {total_labels:,}")
    print(f"  Resilient (1): {resilient_count:,} ({resilient_count/total_labels*100:.1f}%)")
    print(f"  Failed (0): {failed_count:,} ({failed_count/total_labels*100:.1f}%)")
    
    # Create PyG Data objects
    print("\n" + "="*70)
    print("STEP 5: CREATING PYTORCH GEOMETRIC DATA")
    print("="*70)
    
    data_objects = simulator.create_pyg_data_objects(G, node_df, edge_df, scenarios)
    
    # Save scenarios
    print("\n" + "="*70)
    print("STEP 6: SAVING SCENARIOS")
    print("="*70)
    
    output_dir = simulator.save_scenarios(data_objects)
    
    # Final summary
    print("\n" + "="*70)
    print("✅ SCENARIO GENERATION COMPLETE!")
    print("="*70)
    
    print(f"\n📁 Output Directory: {output_dir}/")
    print(f"  ✓ {num_scenarios} scenario files (scenario_XXXXX.pt)")
    print(f"  ✓ metadata.json with feature descriptions")
    
    print("\n🎯 Next Steps:")
    print("  1. Train GNN with: python train_gnn_realistic.py")
    print("  2. Benchmark ML models with: python benchmark_ml_realistic.py")
    print("  3. Compare GNN vs ML performance")
    
    print("\n💡 Expected Results:")
    print("  - GNN should excel (85-95% accuracy)")
    print("  - ML models should struggle (65-75% accuracy)")
    print("  - GNN leverages graph structure to infer hidden production_impact_pct")
    print("  - ML models can't see cascade patterns")


if __name__ == "__main__":
    main()
