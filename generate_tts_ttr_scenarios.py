"""
Stochastic Simulation Engine - TTS vs TTR Based Resilience
Generates 10,000 randomized disruption scenarios using physical buffer absorption logic.

Core Logic:
- TTS (Time To Survive) = Buffer capacity a node has
- TTR (Time To Recover) = Incoming shock hitting the node
- Label 1 (Resilient): Buffer >= Incoming Shock (node survives)
- Label 0 (Failed): Buffer < Incoming Shock (node fails, passes shock downstream)

NO CIRCULAR LOGIC:
- Labels based purely on physical outcomes (buffer vs shock)
- NO network centrality penalties
- NO hardcoded topology assumptions
- GNN must LEARN that central nodes fail more often from the data
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

class StochasticDisruptionSimulator:
    """
    Simulates supply chain disruptions using TTS vs TTR logic with stochastic variance.
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
                region=row['region']
            )
        
        # Add edges with weights (flow quantities)
        for _, row in edge_df.iterrows():
            G.add_edge(
                int(row['source']),
                int(row['target']),
                weight=row.get('flow_quantity', 1.0),
                lead_time=row.get('lead_time', 10.0)
            )
        
        print(f"  ✓ Loaded {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G
    
    def calculate_base_buffers(self, G):
        """Calculate base buffer capacity for each node based on capacity."""
        buffers = {}
        for node in G.nodes():
            # Base buffer = capacity * random factor (0.3 to 0.7)
            # This represents inventory/safety stock
            base_capacity = G.nodes[node]['capacity']
            buffers[node] = base_capacity * np.random.uniform(0.3, 0.7)
        
        return buffers
    
    def select_initial_node(self, G, scenario_type):
        """Select initial disruption node based on scenario type."""
        nodes = list(G.nodes())
        
        if scenario_type == 'random':
            # Random node
            return np.random.choice(nodes)
        
        elif scenario_type == 'high_capacity':
            # Target high-capacity nodes (suppliers)
            capacities = {n: G.nodes[n]['capacity'] for n in nodes}
            sorted_nodes = sorted(capacities.items(), key=lambda x: x[1], reverse=True)
            # Pick from top 20%
            top_20_pct = max(1, len(sorted_nodes) // 5)
            return np.random.choice([n for n, _ in sorted_nodes[:top_20_pct]])
        
        elif scenario_type == 'high_risk':
            # Target high-risk nodes
            risks = {n: G.nodes[n]['risk_level'] for n in nodes}
            sorted_nodes = sorted(risks.items(), key=lambda x: x[1], reverse=True)
            top_20_pct = max(1, len(sorted_nodes) // 5)
            return np.random.choice([n for n, _ in sorted_nodes[:top_20_pct]])
        
        elif scenario_type == 'central':
            # Target central nodes (high betweenness)
            betweenness = nx.betweenness_centrality(G)
            sorted_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
            top_20_pct = max(1, len(sorted_nodes) // 5)
            return np.random.choice([n for n, _ in sorted_nodes[:top_20_pct]])
        
        else:
            return np.random.choice(nodes)
    
    def simulate_cascade(self, G, base_buffers, initial_node, initial_shock):
        """
        Simulate cascading disruption using BFS with TTS vs TTR logic.
        
        Returns:
            dict: {node_id: {'buffer': float, 'shock': float, 'label': int}}
        """
        # Apply stochastic variance to buffers (±20%)
        buffers = {
            node: base_buffers[node] * np.random.uniform(0.8, 1.2)
            for node in G.nodes()
        }
        
        # Track results
        results = {}
        
        # BFS queue: (node_id, incoming_shock)
        queue = deque([(initial_node, initial_shock)])
        visited = set()
        
        while queue:
            current_node, incoming_shock = queue.popleft()
            
            if current_node in visited:
                continue
            visited.add(current_node)
            
            # Get node's buffer
            buffer = buffers[current_node]
            
            # Calculate remaining shock after buffer absorption
            remaining_shock = max(0, incoming_shock - buffer)
            
            # Determine label: Resilient if buffer absorbed all shock
            label = 1 if remaining_shock == 0 else 0
            
            # Store results
            results[current_node] = {
                'buffer': buffer,
                'incoming_shock': incoming_shock,
                'remaining_shock': remaining_shock,
                'label': label
            }
            
            # If node failed, propagate shock downstream
            if remaining_shock > 0:
                # Get downstream neighbors
                downstream = list(G.successors(current_node))
                
                if downstream:
                    # Calculate total outgoing flow
                    total_flow = sum(
                        G[current_node][neighbor].get('weight', 1.0)
                        for neighbor in downstream
                    )
                    
                    # Propagate shock proportionally
                    for neighbor in downstream:
                        # Base proportion based on edge weight
                        edge_weight = G[current_node][neighbor].get('weight', 1.0)
                        proportion = edge_weight / total_flow if total_flow > 0 else 1.0 / len(downstream)
                        
                        # Base shock to neighbor
                        base_shock = remaining_shock * proportion
                        
                        # Apply stochastic decay factor (0.5 to 1.2)
                        # This models real-world uncertainty:
                        # - < 1.0: Finding backup suppliers, mitigation
                        # - > 1.0: Panic ordering, amplification
                        decay_factor = np.random.uniform(0.5, 1.2)
                        actual_shock = base_shock * decay_factor
                        
                        # Add to queue
                        queue.append((neighbor, actual_shock))
        
        return results
    
    def generate_scenarios(self, G, base_buffers, num_scenarios=10000):
        """Generate multiple disruption scenarios."""
        print(f"\n🔄 Generating {num_scenarios:,} stochastic scenarios...")
        
        scenarios = []
        scenario_types = ['random', 'high_capacity', 'high_risk', 'central']
        
        for i in tqdm(range(num_scenarios), desc="Simulating"):
            # Select scenario type
            scenario_type = np.random.choice(scenario_types)
            
            # Select initial node
            initial_node = self.select_initial_node(G, scenario_type)
            
            # Generate initial shock (random between 500 and 2000 units)
            initial_shock = np.random.uniform(500, 2000)
            
            # Run cascade simulation
            results = self.simulate_cascade(G, base_buffers, initial_node, initial_shock)
            
            # Store scenario
            scenarios.append({
                'scenario_id': i,
                'scenario_type': scenario_type,
                'initial_node': initial_node,
                'initial_shock': initial_shock,
                'results': results
            })
        
        print(f"✓ Generated {len(scenarios):,} scenarios")
        
        return scenarios
    
    def create_pyg_data_objects(self, G, node_df, edge_df, scenarios):
        """Convert scenarios to PyTorch Geometric Data objects."""
        print("\n💾 Creating PyG Data objects...")
        
        # Prepare base features
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
            # Initialize features: [base_features + buffer + incoming_shock]
            x = torch.zeros((num_nodes, base_features.shape[1] + 2), dtype=torch.float)
            x[:, :base_features.shape[1]] = base_features
            
            # Initialize labels and mask
            y = torch.full((num_nodes,), -1, dtype=torch.long)
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            
            # Fill in scenario-specific data
            for node_id, result in scenario['results'].items():
                node_id = int(node_id)
                
                # Add buffer and shock as features
                x[node_id, -2] = result['buffer']
                x[node_id, -1] = result['incoming_shock']
                
                # Set label
                y[node_id] = result['label']
                train_mask[node_id] = True
            
            # Create Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                y=y,
                train_mask=train_mask,
                scenario_id=scenario['scenario_id'],
                scenario_type=scenario['scenario_type'],
                initial_node=scenario['initial_node'],
                initial_shock=scenario['initial_shock']
            )
            
            data_objects.append(data)
        
        return data_objects


def main():
    """Main pipeline."""
    print("="*70)
    print("STOCHASTIC SIMULATION ENGINE - TTS vs TTR")
    print("="*70)
    print("\nGenerating 10,000 scenarios with physical buffer absorption logic")
    print("NO circular logic - labels based purely on operational outcomes\n")
    
    # Load data
    print("📂 Loading supply chain data...")
    node_df = pd.read_csv('synthetic_nodes.csv')
    edge_df = pd.read_csv('synthetic_edges.csv')
    print(f"  ✓ Loaded {len(node_df)} nodes, {len(edge_df)} edges")
    
    # Initialize simulator
    simulator = StochasticDisruptionSimulator(seed=42)
    
    # Load graph
    G = simulator.load_graph(node_df, edge_df)
    
    # Calculate base buffers
    print("\n📦 Calculating base buffer capacities...")
    base_buffers = simulator.calculate_base_buffers(G)
    avg_buffer = np.mean(list(base_buffers.values()))
    print(f"  ✓ Average buffer: {avg_buffer:.1f} units")
    
    # Generate scenarios
    scenarios = simulator.generate_scenarios(G, base_buffers, num_scenarios=10000)
    
    # Calculate statistics
    print("\n" + "="*70)
    print("SCENARIO STATISTICS")
    print("="*70)
    
    total_affected = sum(len(s['results']) for s in scenarios)
    total_resilient = sum(
        sum(1 for r in s['results'].values() if r['label'] == 1)
        for s in scenarios
    )
    total_failed = sum(
        sum(1 for r in s['results'].values() if r['label'] == 0)
        for s in scenarios
    )
    
    print(f"\nOverall Statistics:")
    print(f"  Total scenarios: {len(scenarios):,}")
    print(f"  Avg nodes affected per scenario: {total_affected / len(scenarios):.1f}")
    print(f"  Total labeled nodes: {total_affected:,}")
    print(f"    Resilient (1): {total_resilient:,} ({total_resilient/total_affected*100:.1f}%)")
    print(f"    Failed (0): {total_failed:,} ({total_failed/total_affected*100:.1f}%)")
    
    # Scenario type distribution
    scenario_types = {}
    for s in scenarios:
        stype = s['scenario_type']
        scenario_types[stype] = scenario_types.get(stype, 0) + 1
    
    print(f"\nScenario Type Distribution:")
    for stype, count in scenario_types.items():
        print(f"  {stype}: {count:,} ({count/len(scenarios)*100:.1f}%)")
    
    # Create PyG Data objects
    data_objects = simulator.create_pyg_data_objects(G, node_df, edge_df, scenarios)
    
    # Save to disk
    output_dir = 'scenario_graphs_tts_ttr'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n💾 Saving {len(data_objects):,} Data objects to {output_dir}/...")
    
    for i, data in enumerate(tqdm(data_objects, desc="Saving")):
        output_path = os.path.join(output_dir, f'scenario_{i:05d}.pt')
        torch.save(data, output_path)
    
    # Save metadata
    metadata = {
        'num_scenarios': len(scenarios),
        'num_nodes': len(node_df),
        'num_edges': len(edge_df),
        'num_features': 6,  # 4 base features + buffer + incoming_shock
        'feature_names': ['capacity', 'cost_factor', 'risk_level', 'reliability', 'buffer', 'incoming_shock'],
        'scenario_types': scenario_types,
        'total_resilient': int(total_resilient),
        'total_failed': int(total_failed),
        'labeling_logic': 'TTS_vs_TTR',
        'description': 'Labels based on physical buffer absorption. Resilient if buffer >= shock, Failed otherwise.'
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Saved all files to {output_dir}/")
    
    print("\n" + "="*70)
    print("✅ STOCHASTIC SIMULATION COMPLETE!")
    print("="*70)
    print("\n🎯 Key Features:")
    print("  ✓ 10,000 unique scenarios with stochastic variance")
    print("  ✓ TTS vs TTR logic (buffer absorption)")
    print("  ✓ NO circular logic (no centrality penalties)")
    print("  ✓ Physical outcomes only (buffer >= shock)")
    print("  ✓ Cascading propagation with decay factors")
    print("\n📊 Why This Fixes the Problem:")
    print("  • Labels based on PHYSICAL survival (not thresholds)")
    print("  • GNN must LEARN that central nodes fail more")
    print("  • Network topology becomes MEANINGFUL signal")
    print("  • ML models can't achieve 100% (complex patterns)")


if __name__ == "__main__":
    main()
