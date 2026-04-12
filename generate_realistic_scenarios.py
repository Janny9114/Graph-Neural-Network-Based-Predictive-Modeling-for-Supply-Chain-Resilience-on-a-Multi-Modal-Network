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
        
        # Add nodes with attributes (including x, y coordinates)
        for idx, row in node_df.iterrows():
            G.add_node(
                idx,
                capacity=row['capacity'],
                risk_level=row['risk_level'],
                tier=row['tier'],
                region=row['region'],
                cost_factor=row['cost_factor'],
                reliability=row['reliability'],
                x=row['x'],  # Add x coordinate
                y=row['y']   # Add y coordinate
            )
        
        # Add edges with weights (use capacity_share as weight)
        for idx, row in edge_df.iterrows():
            G.add_edge(
                row['source'],
                row['target'],
                weight=row.get('weight', row.get('capacity_share', 1.0))
            )
        
        print(f"  ✓ Loaded {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Analyze spectral radius for cascade stability
        self.analyze_spectral_radius(G)
        
        return G
    
    def analyze_spectral_radius(self, G):
        """
        Analyze spectral radius ρ(W) to determine cascade stability.
        
        ρ(W) > 1: Cascades AMPLIFY (unstable system)
        ρ(W) < 1: Cascades DAMPEN (stable system)
        ρ(W) ≈ 1: CRITICAL (just-in-time systems)
        """
        print("\n📊 Spectral Radius Analysis:")
        
        # Build weighted adjacency matrix
        W = nx.to_numpy_array(G, weight='weight')
        
        # Normalize by out-degree (row-stochastic matrix)
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        W_normalized = W / row_sums
        
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvals(W_normalized)
        rho = np.max(np.abs(eigenvalues))
        
        print(f"  Spectral radius ρ(W) = {rho:.4f}")
        
        if rho > 1.05:
            print(f"  ⚠ UNSTABLE: Cascades will AMPLIFY by {(rho-1)*100:.1f}%")
            print(f"  → System prone to runaway failures")
            self.cascade_regime = 'amplifying'
        elif rho < 0.95:
            print(f"  ✓ STABLE: Cascades will DAMPEN by {(1-rho)*100:.1f}%")
            print(f"  → System absorbs shocks well")
            self.cascade_regime = 'dampening'
        else:
            print(f"  ⚠ CRITICAL: System near instability threshold")
            print(f"  → Just-in-time optimization (high risk)")
            self.cascade_regime = 'critical'
        
        # Store for use in decay factor calibration
        self.spectral_radius = rho
        
        return rho
    
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
    
    def simulate_multi_node_cascade(self, G, base_buffers, initial_nodes, initial_impact_pcts):
        """
        Simulate cascading disruption from multiple initial nodes simultaneously.
        
        Args:
            G: NetworkX graph
            base_buffers: Dict of base buffer values for each node
            initial_nodes: List of nodes where disruptions start
            initial_impact_pcts: List of initial production impacts for each node
        
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
        # Initialize with all initial nodes
        queue = deque([(node, impact) for node, impact in zip(initial_nodes, initial_impact_pcts)])
        
        while queue:
            current_node, impact_pct = queue.popleft()
            
            if current_node in visited:
                # If already visited, accumulate impact
                if current_node in results:
                    results[current_node]['production_impact_pct'] += impact_pct
                    results[current_node]['production_impact_units'] = (
                        G.nodes[current_node]['capacity'] * results[current_node]['production_impact_pct']
                    )
                    # Recalculate remaining impact
                    remaining_impact = max(0, results[current_node]['production_impact_units'] - results[current_node]['buffer'])
                    results[current_node]['remaining_impact'] = remaining_impact
                    results[current_node]['label'] = 1 if remaining_impact == 0 else 0
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
            is_initial = 1 if current_node in initial_nodes else 0
            results[current_node] = {
                'buffer': buffer,
                'production_impact_pct': impact_pct,
                'production_impact_units': impact_units,
                'remaining_impact': remaining_impact,
                'label': label,
                'is_disrupted': is_initial
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
                        decay_factor = np.random.uniform(0.5, 1.2)
                        actual_impact_units = base_impact_units * decay_factor
                        
                        # Convert back to percentage for neighbor
                        neighbor_capacity = G.nodes[neighbor]['capacity']
                        neighbor_impact_pct = actual_impact_units / neighbor_capacity
                        
                        # Add to queue
                        queue.append((neighbor, neighbor_impact_pct))
        
        return results
    
    def select_regional_cluster_variable_radius(self, G):
        """
        Select nodes in same geographic region with VARIABLE radius.
        Simulates different disaster types with different impact areas.
        
        NOW WORKS WITH Z-SCORE NORMALIZED COORDINATES!
        Coordinates are Z-score normalized (mean=0, std=1, range≈-3 to +3)
        
        Disaster Types:
        - Small (0.3-0.8 std):   Factory fire, local power outage (10-25% of nodes)
        - Medium (0.8-1.5 std):  Severe storm, regional flooding (25-50% of nodes)
        - Large (1.5-2.5 std):   Hurricane, major earthquake (50-75% of nodes)
        - Massive (2.5-4.0 std): Tsunami, widespread disaster (75-95% of nodes)
        """
        nodes = list(G.nodes())
        
        # Pick random epicenter
        epicenter = np.random.choice(nodes)
        epicenter_x = G.nodes[epicenter]['x']  # Z-score normalized
        epicenter_y = G.nodes[epicenter]['y']  # Z-score normalized
        
        # VARIABLE RADIUS based on disaster type - NOW IN Z-SCORE UNITS!
        disaster_type = np.random.choice(
            ['small', 'medium', 'large', 'massive'],
            p=[0.25, 0.35, 0.30, 0.10]
        )
        
        # Set radius in Z-score units (typical coordinate range: -3 to +3)
        if disaster_type == 'small':
            radius = np.random.uniform(0.3, 0.8)    # ~10-25% of nodes
            impact_severity = np.random.uniform(0.4, 0.7)
        elif disaster_type == 'medium':
            radius = np.random.uniform(0.8, 1.5)    # ~25-50% of nodes
            impact_severity = np.random.uniform(0.6, 0.85)
        elif disaster_type == 'large':
            radius = np.random.uniform(1.5, 2.5)    # ~50-75% of nodes
            impact_severity = np.random.uniform(0.7, 0.95)
        else:  # massive
            radius = np.random.uniform(2.5, 4.0)    # ~75-95% of nodes
            impact_severity = np.random.uniform(0.8, 1.0)
        
        # Find all nodes within radius
        regional_nodes = []
        distances = []
        
        for node in nodes:
            node_x = G.nodes[node]['x']
            node_y = G.nodes[node]['y']
            
            # Calculate Euclidean distance
            distance = np.sqrt((node_x - epicenter_x)**2 + (node_y - epicenter_y)**2)
            
            if distance <= radius:
                regional_nodes.append(node)
                distances.append(distance)
        
        # If too few nodes, expand radius
        if len(regional_nodes) < 3:
            radius *= 1.5
            regional_nodes = []
            distances = []
            for node in nodes:
                node_x = G.nodes[node]['x']
                node_y = G.nodes[node]['y']
                distance = np.sqrt((node_x - epicenter_x)**2 + (node_y - epicenter_y)**2)
                if distance <= radius:
                    regional_nodes.append(node)
                    distances.append(distance)
        
        # Select affected nodes (closer nodes more likely)
        num_affected = min(
            len(regional_nodes),
            np.random.randint(max(2, len(regional_nodes)//3), len(regional_nodes)+1)
        )
        
        # Calculate selection probabilities based on distance
        max_distance = max(distances) if distances else 1
        probabilities = [1 - (d / max_distance) * 0.7 for d in distances]
        probabilities = np.array(probabilities) / sum(probabilities)
        
        selected_indices = np.random.choice(
            len(regional_nodes),
            size=num_affected,
            replace=False,
            p=probabilities
        )
        selected_nodes = [regional_nodes[i] for i in selected_indices]
        selected_distances = [distances[i] for i in selected_indices]
        
        return {
            'nodes': selected_nodes,
            'epicenter': epicenter,
            'radius': radius,
            'disaster_type': disaster_type,
            'impact_severity': impact_severity,
            'distances': selected_distances
        }
    
    def simulate_regional_failure_variable(self, G, base_buffers):
        """
        Simulate regional disaster with variable radius and distance-based impact.
        """
        cluster_info = self.select_regional_cluster_variable_radius(G)
        
        initial_nodes = cluster_info['nodes']
        radius = cluster_info['radius']
        base_severity = cluster_info['impact_severity']
        distances = cluster_info['distances']
        
        # Calculate impact for each node based on distance from epicenter
        initial_impact_pcts = []
        for distance in distances:
            # Impact decreases with distance (exponential decay)
            distance_factor = np.exp(-distance / (radius * 0.5))
            
            # Node impact = base severity × distance factor × random variance
            node_impact = base_severity * distance_factor * np.random.uniform(0.85, 1.15)
            node_impact = np.clip(node_impact, 0.1, 1.0)
            
            initial_impact_pcts.append(node_impact)
        
        # Simulate cascade
        results = self.simulate_multi_node_cascade(
            G, base_buffers, initial_nodes, initial_impact_pcts
        )
        
        return {
            'scenario_type': 'regional_failure_variable',
            'initial_node': initial_nodes,
            'initial_impact_pct': initial_impact_pcts,
            'epicenter': cluster_info['epicenter'],
            'radius': radius,
            'disaster_type': cluster_info['disaster_type'],
            'results': results
        }
    
    def select_port_nodes(self, G, num_ports=2):
        """Select high-degree nodes representing ports/hubs."""
        out_degrees = dict(G.out_degree())
        sorted_nodes = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)
        top_20_pct = max(1, len(sorted_nodes) // 5)
        port_candidates = [n for n, _ in sorted_nodes[:top_20_pct]]
        selected_ports = np.random.choice(port_candidates, size=min(num_ports, len(port_candidates)), replace=False)
        return list(selected_ports)
    
    def simulate_port_congestion(self, G, base_buffers):
        """
        Simulate port congestion affecting all downstream nodes.
        Lead time increases propagate through supply chain.
        """
        num_ports = np.random.randint(1, 4)
        port_nodes = self.select_port_nodes(G, num_ports=num_ports)
        port_impacts = [np.random.uniform(0.3, 0.6) for _ in port_nodes]
        
        results = self.simulate_multi_node_cascade(
            G, base_buffers, port_nodes, port_impacts
        )
        
        return {
            'scenario_type': 'port_congestion',
            'initial_node': port_nodes,
            'initial_impact_pct': port_impacts,
            'results': results
        }
    
    def generate_scenarios(self, G, base_buffers, num_scenarios=10000):
        """Generate multiple disruption scenarios with new types."""
        print(f"\n🎲 Generating {num_scenarios} disruption scenarios...")
        
        scenarios = []
        # NEW: 7 scenario types with regional failures and port congestion
        scenario_types = [
            'regional_failure_variable',  # 28% - Variable radius regional
            'regional_failure_variable',  # 28% - Variable radius regional
            'port_congestion',            # 14% - Lead time delays
            'multi_node_10',              # 14% - Complex multi-source
            'central',                    # 7% - High centrality
            'high_risk',                  # 7% - High risk zones
            'random'                      # 2% - Random
        ]
        
        for i in tqdm(range(num_scenarios), desc="Simulating scenarios"):
            # Select scenario type
            scenario_type = scenario_types[i % len(scenario_types)]
            
            if scenario_type == 'regional_failure_variable':
                # Regional disaster with variable radius
                scenario = self.simulate_regional_failure_variable(G, base_buffers)
                scenario['scenario_id'] = i
                scenarios.append(scenario)
                
            elif scenario_type == 'port_congestion':
                # Port congestion scenario
                scenario = self.simulate_port_congestion(G, base_buffers)
                scenario['scenario_id'] = i
                scenarios.append(scenario)
                
            elif scenario_type == 'multi_node_10':
                # Complex scenario: 10 random nodes disrupted simultaneously
                nodes = list(G.nodes())
                initial_nodes = list(np.random.choice(nodes, size=10, replace=False))
                initial_impact_pcts = [np.random.uniform(0.1, 1.0) for _ in range(10)]
                
                # Simulate multi-node cascade
                results = self.simulate_multi_node_cascade(G, base_buffers, initial_nodes, initial_impact_pcts)
                
                scenarios.append({
                    'scenario_id': i,
                    'scenario_type': scenario_type,
                    'initial_node': initial_nodes,  # List of nodes
                    'initial_impact_pct': initial_impact_pcts,  # List of impacts
                    'results': results
                })
            else:
                # Single-node disruption scenarios
                initial_node = self.select_initial_node(G, scenario_type)
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
        
        # Count scenario types
        multi_node_count = sum(1 for s in scenarios if s['scenario_type'] == 'multi_node_10')
        regional_count = sum(1 for s in scenarios if 'regional_failure' in s['scenario_type'])
        port_count = sum(1 for s in scenarios if s['scenario_type'] == 'port_congestion')
        
        print(f"\n  ✓ Generated {num_scenarios} scenarios")
        print(f"  ✓ Average nodes affected per scenario: {avg_affected:.1f}")
        print(f"  ✓ Regional failure scenarios: {regional_count} ({regional_count/num_scenarios*100:.1f}%)")
        print(f"  ✓ Port congestion scenarios: {port_count} ({port_count/num_scenarios*100:.1f}%)")
        print(f"  ✓ Multi-node (10 disruptions) scenarios: {multi_node_count} ({multi_node_count/num_scenarios*100:.1f}%)")
        
        # Regional failure statistics
        if regional_count > 0:
            regional_scenarios = [s for s in scenarios if 'regional_failure' in s['scenario_type']]
            disaster_types = {}
            for s in regional_scenarios:
                if 'disaster_type' in s:
                    dtype = s['disaster_type']
                    disaster_types[dtype] = disaster_types.get(dtype, 0) + 1
            
            print(f"\n  📊 Regional Failure Breakdown:")
            print(f"    Small disasters (10-50 km):   {disaster_types.get('small', 0)}")
            print(f"    Medium disasters (50-150 km):  {disaster_types.get('medium', 0)}")
            print(f"    Large disasters (150-300 km):  {disaster_types.get('large', 0)}")
            print(f"    Massive disasters (300-500 km): {disaster_types.get('massive', 0)}")
        
        return scenarios
    
    def create_pyg_data_objects(self, G, node_df, edge_df, scenarios):
        """
        Convert scenarios to PyTorch Geometric Data objects.
        
        Feature design (HARDER VERSION):
        - ALL nodes get buffer (realistic - companies know their inventory)
        - NO production_impact_pct (hidden from everyone)
        - NO is_disrupted flag (must infer from graph)
        - Includes x,y coordinates for spatial analysis
        - ALL FEATURES NORMALIZED with StandardScaler
        
        Features (7 total):
        [0] capacity (normalized)
        [1] cost_factor (normalized)
        [2] risk_level (normalized)
        [3] reliability (normalized)
        [4] x (longitude, normalized)
        [5] y (latitude, normalized)
        [6] buffer (normalized per scenario)
        
        GNN must use graph structure to infer:
        - Which node is disrupted
        - Impact magnitude
        - Cascade patterns
        """
        print("\n💾 Creating PyG Data objects with NORMALIZED features...")
        
        # Prepare base features (first 6 features including x,y)
        from sklearn.preprocessing import StandardScaler
        
        base_features_raw = node_df[['capacity', 'cost_factor', 'risk_level', 'reliability', 'x', 'y']].values
        
        # Normalize base features with StandardScaler
        scaler = StandardScaler()
        base_features_normalized = scaler.fit_transform(base_features_raw)
        base_features = torch.tensor(base_features_normalized, dtype=torch.float)
        
        print(f"  ✓ Normalized base features with StandardScaler")
        print(f"    - Mean: {base_features.mean(dim=0).numpy()}")
        print(f"    - Std:  {base_features.std(dim=0).numpy()}")
        
        # Calculate buffer statistics for normalization
        all_buffers = []
        for scenario in scenarios:
            for result in scenario['results'].values():
                all_buffers.append(result['buffer'])
        
        buffer_mean = np.mean(all_buffers)
        buffer_std = np.std(all_buffers)
        print(f"  ✓ Buffer normalization: mean={buffer_mean:.2f}, std={buffer_std:.2f}")
        
        # Create edge_index
        edge_index = torch.tensor([
            edge_df['source'].values,
            edge_df['target'].values
        ], dtype=torch.long)
        
        num_nodes = len(node_df)
        data_objects = []
        
        for scenario in tqdm(scenarios, desc="Creating Data objects"):
            # Initialize features: [base_features + buffer] (7 features total)
            x = torch.zeros((num_nodes, base_features.shape[1] + 1), dtype=torch.float)
            x[:, :base_features.shape[1]] = base_features
            
            # Initialize labels and mask
            y = torch.full((num_nodes,), -1, dtype=torch.long)
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            
            # Fill in scenario-specific data
            for node_id, result in scenario['results'].items():
                node_id = int(node_id)
                
                # Feature 6: buffer (ALL nodes know this) - index 6 because we have capacity, cost, risk, reliability, x, y, buffer
                x[node_id, 6] = result['buffer']
                
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
            
            # Add metadata (convert lists to tensors for multi-node scenarios)
            data.scenario_id = scenario['scenario_id']
            data.scenario_type = scenario['scenario_type']
            
            # Handle initial_node and initial_impact_pct (can be scalar or list)
            if isinstance(scenario['initial_node'], list):
                # Multi-node scenario: store as tensor
                data.initial_nodes = torch.tensor(scenario['initial_node'], dtype=torch.long)
                data.initial_impacts = torch.tensor(scenario['initial_impact_pct'], dtype=torch.float)
                data.num_initial_disruptions = len(scenario['initial_node'])
            else:
                # Single-node scenario: store as scalar
                data.initial_nodes = torch.tensor([scenario['initial_node']], dtype=torch.long)
                data.initial_impacts = torch.tensor([scenario['initial_impact_pct']], dtype=torch.float)
                data.num_initial_disruptions = 1
            
            data_objects.append(data)
        
        print(f"  ✓ Created {len(data_objects)} Data objects")
        print(f"  ✓ Feature dimensions: {data_objects[0].x.shape}")
        print(f"  ✓ Features: [capacity, cost_factor, risk_level, reliability, x, y, buffer]")
        
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
                'x',
                'y',
                'buffer'
            ],
            'feature_description': {
                'capacity': 'Node production capacity (all nodes)',
                'cost_factor': 'Operational cost factor (all nodes)',
                'risk_level': 'Baseline vulnerability (all nodes)',
                'reliability': 'Historical performance (all nodes)',
                'x': 'Longitude coordinate (geographic location)',
                'y': 'Latitude coordinate (geographic location)',
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


def load_preprocessed_graph(graph_path='supply_chain_graph.pt'):
    """
    Load preprocessed graph from graph_construction.py output.
    Extracts node attributes and edge information.
    """
    print(f"\n📊 Loading preprocessed graph from {graph_path}...")
    
    # Load PyG Data object
    data = torch.load(graph_path, weights_only=False)
    
    print(f"  ✓ Loaded graph with {data.num_nodes} nodes, {data.num_edges} edges")
    print(f"  ✓ Feature dimensions: {data.x.shape}")
    
    # Extract node attributes from features
    # Assuming features are: [tier_onehot(4), capacity, cost_factor, risk_level, reliability, x, y]
    # Total: 4 + 6 = 10 features
    
    num_nodes = data.num_nodes
    
    # Create node_df from features
    node_data = {
        'node_id': list(range(num_nodes)),
        'capacity': data.x[:, 4].numpy(),      # Feature index 4
        'cost_factor': data.x[:, 5].numpy(),   # Feature index 5
        'risk_level': data.x[:, 6].numpy(),    # Feature index 6
        'reliability': data.x[:, 7].numpy(),   # Feature index 7
        'x': data.x[:, 8].numpy(),             # Feature index 8 (normalized)
        'y': data.x[:, 9].numpy(),             # Feature index 9 (normalized)
    }
    
    # Extract tier from one-hot encoding
    tier_onehot = data.x[:, :4].numpy()
    node_data['tier'] = tier_onehot.argmax(axis=1)
    
    # Add placeholder region (not critical for simulation)
    node_data['region'] = ['Region_' + str(i % 5) for i in range(num_nodes)]
    
    node_df = pd.DataFrame(node_data)
    
    # Create edge_df from edge_index
    num_edges = data.edge_index.shape[1]
    edge_data = {
        'source': data.edge_index[0].numpy(),
        'target': data.edge_index[1].numpy(),
    }
    
    # Add edge weights if available
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        edge_weights = data.edge_attr.numpy()
        # Handle different edge_attr shapes
        if len(edge_weights.shape) > 1:
            edge_weights = edge_weights[:, 0]  # Take first column if multi-dimensional
        edge_data['weight'] = edge_weights
    else:
        edge_data['weight'] = np.ones(num_edges)
    
    edge_df = pd.DataFrame(edge_data)
    
    print(f"  ✓ Extracted {len(node_df)} nodes with attributes")
    print(f"  ✓ Extracted {len(edge_df)} edges")
    print(f"  ✓ Coordinates are already normalized from preprocessing!")
    
    return node_df, edge_df


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
    print("  ✓ Uses preprocessed data from graph_construction.py")
    
    # Load preprocessed data
    print("\n" + "="*70)
    print("STEP 1: LOADING PREPROCESSED DATA")
    print("="*70)
    
    node_df, edge_df = load_preprocessed_graph('supply_chain_graph.pt')
    
    print(f"  ✓ Loaded {len(node_df)} nodes")
    print(f"  ✓ Loaded {len(edge_df)} edges")
    print(f"  ✓ Using normalized coordinates from preprocessing")
    
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
