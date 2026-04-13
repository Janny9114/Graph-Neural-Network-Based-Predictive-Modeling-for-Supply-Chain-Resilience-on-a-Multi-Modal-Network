"""
Paper-Based Scenario Generator
Implements the 5 disruption scenarios from:
"Graph Neural Network-Based Predictive Modeling for Enhanced Supply Chain Resilience 
against Multi-Modal Disruptions"

Scenarios:
1. Random Supplier Failure: 10 random suppliers
2. High-Capacity Supplier Failure: 5 highest-capacity suppliers
3. High-Risk Manufacturer Failure: 8 high-risk manufacturers
4. Central Distributor Failure: 4 distributors with highest betweenness centrality
5. Multiple Transportation Delays: 25 random edges

Node disruptions: Reliability reduced by 70%
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


class PaperBasedDisruptionSimulator:
    """
    Implements disruption scenarios exactly as described in the paper.
    """
    
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
        
    def load_graph(self, node_df, edge_df):
        """Load graph from preprocessed data."""
        print("\n📊 Building NetworkX graph...")
        
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for idx, row in node_df.iterrows():
            G.add_node(
                idx,
                capacity=row['capacity'],
                cost_factor=row['cost_factor'],
                risk_level=row['risk_level'],
                reliability=row['reliability'],
                tier=row['tier'],
                region=row['region'],
                x=row['x'],
                y=row['y']
            )
        
        # Add edges with weights
        for _, row in edge_df.iterrows():
            G.add_edge(
                row['source'],
                row['target'],
                weight=row.get('weight', 1.0)
            )
        
        print(f"  ✓ Nodes: {G.number_of_nodes()}")
        print(f"  ✓ Edges: {G.number_of_edges()}")
        
        return G
    
    def simulate_node_disruption_cascade(self, G, disrupted_nodes, reliability_reduction=0.7, risk_increase=1.0):
        """
        Simulate cascade from node disruptions using PAPER METHODOLOGY.
        
        Paper Equations (60-68):
        1) Node-based disruptions: Modify reliability and risk of affected nodes
           r'_j = θ_r · r_j  where θ_r ∈ V_disrupted  (Eq. 60)
           ρ'_j = min(θ_ρ · r_j, 1.0)  where θ_ρ ∈ V_disrupted  (Eq. 61)
           
           where θ_r < 1 is reliability reduction factor (0.3 for 70% reduction)
           and θ_ρ > 1 is risk increase factor (2.0 for doubling risk)
        
        2) Edge-based disruptions: Modify lead time of affected edges
           l'_ij = θ_l · l_ij  where (v_i, v_j) ∈ E_disrupted  (Eq. 62)
           
           where θ_l > 1 is lead time increase factor (3.0 for 3x delay)
        
        3) Impact Quantification (Eq. 63-68):
           - Total nodes affected
           - Resilient to vulnerable transitions: R→V = Σ I(y'_j = 1 ∧ y_j = 0)
           - Vulnerable to resilient transitions: V→R = Σ I(y'_j = 0 ∧ y_j = 1)
           - Resilience reduction percentage: RRP = [Σ I(y'_j = 1) / Σ I(y_j = 1)] × 100%
        
        Args:
            G: NetworkX graph
            disrupted_nodes: List of initially disrupted node IDs
            reliability_reduction: θ_r factor (default 0.7 = 70% reduction → θ_r = 0.3)
            risk_increase: θ_ρ factor (default 1.0 = no increase → θ_ρ = 2.0)
        
        Returns:
            Dict with results for each affected node
        """
        results = {}
        
        # Paper Eq. 60-61: Apply disruption to affected nodes
        # θ_r = 1 - reliability_reduction (e.g., 0.7 reduction → θ_r = 0.3)
        theta_r = 1.0 - reliability_reduction
        theta_rho = 1.0 + risk_increase  # θ_ρ = 2.0 for doubling risk
        
        # Store modified reliability and risk for all nodes
        modified_reliability = {}
        modified_risk = {}
        
        for node in G.nodes():
            if node in disrupted_nodes:
                # Apply disruption: r'_j = θ_r · r_j
                modified_reliability[node] = theta_r * G.nodes[node]['reliability']
                # Apply risk increase: ρ'_j = min(θ_ρ · ρ_j, 1.0)
                modified_risk[node] = min(theta_rho * G.nodes[node]['risk_level'], 1.0)
            else:
                modified_reliability[node] = G.nodes[node]['reliability']
                modified_risk[node] = G.nodes[node]['risk_level']
        
        # Calculate impact score based on geographical proximity and node vulnerability
        # Paper: imp_v,d = s_d·exp(-dist((l_v, l_d), (x_d, y_d))^2 / (2·σ^2·r_d))
        impact_scores = {}
        
        for node in G.nodes():
            if node in disrupted_nodes:
                impact_scores[node] = 1.0  # Directly disrupted
            else:
                # Calculate impact from all disrupted nodes
                total_impact = 0.0
                for disrupted_node in disrupted_nodes:
                    # Euclidean distance
                    dx = G.nodes[node]['x'] - G.nodes[disrupted_node]['x']
                    dy = G.nodes[node]['y'] - G.nodes[disrupted_node]['y']
                    dist = np.sqrt(dx**2 + dy**2)
                    
                    # Severity factor (based on disrupted node's importance)
                    severity = G.nodes[disrupted_node]['capacity']
                    
                    # Vulnerability factor (based on current node's reliability)
                    vulnerability = 1.0 - G.nodes[node]['reliability']
                    
                    # Impact calculation (Gaussian decay with distance)
                    sigma = 1.0  # Standard deviation for spatial decay
                    impact = severity * np.exp(-dist**2 / (2 * sigma**2 * (1.0 - vulnerability + 0.1)))
                    total_impact += impact
                
                impact_scores[node] = min(total_impact, 1.0)
        
        # Propagate cascade through supply chain
        queue = deque(disrupted_nodes)
        visited = set(disrupted_nodes)
        
        while queue:
            current_node = queue.popleft()
            
            # Propagate to downstream nodes
            for successor in G.successors(current_node):
                if successor in visited:
                    continue
                
                # Calculate supply shortage
                predecessors = list(G.predecessors(successor))
                if len(predecessors) == 0:
                    continue
                
                # Calculate average health ratio of suppliers (how degraded they are)
                # Health ratio = modified_reliability / original_reliability
                avg_supplier_health = np.mean([
                    modified_reliability[pred] / (G.nodes[pred]['reliability'] + 1e-8) 
                    for pred in predecessors
                ])
                
                # Node's operational capacity depends on supplier health
                # Only degrade if suppliers are actually degraded (health < 1.0)
                operational_capacity = modified_reliability[successor] * avg_supplier_health
                
                # If significantly degraded, propagate cascade
                if operational_capacity < G.nodes[successor]['reliability'] * 0.9:
                    modified_reliability[successor] = operational_capacity
                    impact_scores[successor] = max(impact_scores[successor], 1.0 - operational_capacity)
                    visited.add(successor)
                    queue.append(successor)
        
        # Paper Eq. 63-64: Compute resilience labels
        # y'_j = arg max_c[z'_j] on modified graph G'
        # CRITICAL FIX: Label ALL nodes, not just disrupted ones!
        
        for node_id in G.nodes():
            if node_id in visited:
                # Node was affected by disruption
                impact = impact_scores[node_id]
                original_reliability = G.nodes[node_id]['reliability']
                current_reliability = modified_reliability[node_id]
                
                # Calculate resilience metrics
                # Resilience = ability to maintain function under disruption
                resilience_score = current_reliability / (original_reliability + 1e-8)
                
                # Paper uses binary classification (resilient vs vulnerable)
                # We extend to 3-class for more granularity
                # Adjusted thresholds for 70% reliability reduction (θ_r = 0.3):
                # - High resilience (>0.5): Class 2 (Normal) - maintains >50% function
                # - Medium resilience (0.25-0.5): Class 1 (Degraded) - maintains 25-50% function
                # - Low resilience (<0.25): Class 0 (Failed) - maintains <25% function
                
                if resilience_score >= 0.7:
                    label = 2  # Normal (resilient)
                elif resilience_score >= 0.25:
                    label = 1  # Degraded (partially vulnerable)
                else:
                    label = 0  # Failed (vulnerable)
                
                results[node_id] = {
                    'original_reliability': original_reliability,
                    'modified_reliability': current_reliability,
                    'resilience_score': resilience_score,
                    'impact_score': impact,
                    'label': label,
                    'is_initially_disrupted': 1 if node_id in disrupted_nodes else 0
                }
            else:
                # Node was NOT affected - label as Normal (class 2)
                results[node_id] = {
                    'original_reliability': G.nodes[node_id]['reliability'],
                    'modified_reliability': G.nodes[node_id]['reliability'],
                    'resilience_score': 1.0,
                    'impact_score': 0.0,
                    'label': 2,  # Normal (unaffected)
                    'is_initially_disrupted': 0
                }
        
        return results
    
    def simulate_edge_disruption_cascade(self, G, disrupted_edges, delay_factor=2.0):
        """
        Simulate cascade from edge disruptions (transportation delays).
        
        Paper methodology:
        - Disrupted edges have increased lead time
        - Downstream nodes affected by supply delays
        
        Args:
            G: NetworkX graph
            disrupted_edges: List of (source, target) tuples
            delay_factor: Multiplier for lead time (default 2.0 = 2x delay)
        
        Returns:
            Dict with results for each affected node
        """
        results = {}
        
        # Track which nodes are affected by delays
        affected_nodes = set()
        for source, target in disrupted_edges:
            affected_nodes.add(target)
        
        # Propagate delays downstream
        queue = deque(affected_nodes)
        visited = set(affected_nodes)
        delay_impact = {node: 1.0 for node in affected_nodes}  # 1.0 = full delay impact
        
        while queue:
            current_node = queue.popleft()
            
            # Propagate to downstream nodes (with attenuation)
            for successor in G.successors(current_node):
                if successor not in visited:
                    # Delay impact attenuates as it propagates
                    propagated_impact = delay_impact[current_node] * 0.7
                    
                    if propagated_impact > 0.2:  # Only propagate if significant
                        delay_impact[successor] = propagated_impact
                        visited.add(successor)
                        queue.append(successor)
        
        # Generate results with 3-class labels
        # CRITICAL FIX: Label ALL nodes, not just affected ones!
        for node_id in G.nodes():
            if node_id in delay_impact:
                # Node was affected by delays
                impact = delay_impact[node_id]
                
                # 3-class labeling based on delay impact
                # Adjusted thresholds for better balance
                if impact >= 0.6:
                    label = 0  # Failed (severe delays)
                elif impact >= 0.3:
                    label = 1  # Degraded (moderate delays)
                else:
                    label = 2  # Normal (minor delays)
                
                results[node_id] = {
                    'delay_impact': impact,
                    'label': label,
                    'is_directly_affected': 1 if node_id in affected_nodes else 0
                }
            else:
                # Node was NOT affected - label as Normal (class 2)
                results[node_id] = {
                    'delay_impact': 0.0,
                    'label': 2,  # Normal (unaffected)
                    'is_directly_affected': 0
                }
        
        return results
    
    def scenario_1_random_supplier_failure(self, G):
        """
        Scenario 1: Random Supplier Failure
        - 10 random suppliers (tier 0)
        - Reliability reduced by 70%
        """
        # Find all supplier nodes
        supplier_nodes = [n for n in G.nodes() if G.nodes[n]['tier'] == 0]
        
        if len(supplier_nodes) < 10:
            print(f"  ⚠ Warning: Only {len(supplier_nodes)} suppliers available")
            num_to_select = len(supplier_nodes)
        else:
            num_to_select = 10
        
        # Select 10 random suppliers
        disrupted_nodes = list(np.random.choice(supplier_nodes, size=num_to_select, replace=False))
        
        # Simulate cascade
        results = self.simulate_node_disruption_cascade(G, disrupted_nodes, reliability_reduction=0.7)
        
        return {
            'scenario_type': 'random_supplier_failure',
            'disrupted_nodes': disrupted_nodes,
            'reliability_reduction': 0.7,
            'results': results
        }
    
    def scenario_2_high_capacity_supplier_failure(self, G):
        """
        Scenario 2: High-Capacity Supplier Failure
        - 5 suppliers with highest capacity
        - Reliability reduced by 70%
        """
        # Find all supplier nodes
        supplier_nodes = [n for n in G.nodes() if G.nodes[n]['tier'] == 0]
        
        # Sort by capacity
        suppliers_by_capacity = sorted(
            [(n, G.nodes[n]['capacity']) for n in supplier_nodes],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Select top 5
        num_to_select = min(5, len(suppliers_by_capacity))
        disrupted_nodes = [n for n, _ in suppliers_by_capacity[:num_to_select]]
        
        # Simulate cascade
        results = self.simulate_node_disruption_cascade(G, disrupted_nodes, reliability_reduction=0.7)
        
        return {
            'scenario_type': 'high_capacity_supplier_failure',
            'disrupted_nodes': disrupted_nodes,
            'reliability_reduction': 0.7,
            'results': results
        }
    
    def scenario_3_high_risk_manufacturer_failure(self, G):
        """
        Scenario 3: High-Risk Manufacturer Failure
        - 8 manufacturers with highest risk scores
        - Reliability reduced by 70%
        """
        # Find all manufacturer nodes (tier 1)
        manufacturer_nodes = [n for n in G.nodes() if G.nodes[n]['tier'] == 1]
        
        # Sort by risk level (highest risk first)
        manufacturers_by_risk = sorted(
            [(n, G.nodes[n]['risk_level']) for n in manufacturer_nodes],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Select top 8
        num_to_select = min(8, len(manufacturers_by_risk))
        disrupted_nodes = [n for n, _ in manufacturers_by_risk[:num_to_select]]
        
        # Simulate cascade
        results = self.simulate_node_disruption_cascade(G, disrupted_nodes, reliability_reduction=0.7)
        
        return {
            'scenario_type': 'high_risk_manufacturer_failure',
            'disrupted_nodes': disrupted_nodes,
            'reliability_reduction': 0.7,
            'results': results
        }
    
    def scenario_4_central_distributor_failure(self, G):
        """
        Scenario 4: Central Distributor Failure
        - 4 distributors with highest betweenness centrality
        - Reliability reduced by 70%
        """
        # Find all distributor nodes (tier 2)
        distributor_nodes = [n for n in G.nodes() if G.nodes[n]['tier'] == 2]
        
        if len(distributor_nodes) == 0:
            print("  ⚠ Warning: No distributor nodes found, using all nodes")
            distributor_nodes = list(G.nodes())
        
        # Calculate betweenness centrality
        #print("  → Calculating betweenness centrality...")
        betweenness = nx.betweenness_centrality(G)
        
        # Filter to distributors and sort by betweenness
        distributors_by_betweenness = sorted(
            [(n, betweenness[n]) for n in distributor_nodes],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Select top 4
        num_to_select = min(4, len(distributors_by_betweenness))
        disrupted_nodes = [n for n, _ in distributors_by_betweenness[:num_to_select]]
        
        # Simulate cascade
        results = self.simulate_node_disruption_cascade(G, disrupted_nodes, reliability_reduction=0.7)
        
        return {
            'scenario_type': 'central_distributor_failure',
            'disrupted_nodes': disrupted_nodes,
            'reliability_reduction': 0.7,
            'results': results
        }
    
    def scenario_5_multiple_transportation_delays(self, G):
        """
        Scenario 5: Multiple Transportation Delays
        - 25 random edges
        - Lead time increased (2x delay)
        """
        edges = list(G.edges())
        
        # Select 25 random edges
        num_to_select = min(25, len(edges))
        disrupted_edges = [edges[i] for i in np.random.choice(len(edges), size=num_to_select, replace=False)]
        
        # Simulate cascade
        results = self.simulate_edge_disruption_cascade(G, disrupted_edges, delay_factor=2.0)
        
        return {
            'scenario_type': 'multiple_transportation_delays',
            'disrupted_edges': disrupted_edges,
            'delay_factor': 2.0,
            'results': results
        }
    
    def generate_paper_scenarios(self, G, num_scenarios=1000):
        """
        Generate scenarios using the 5 paper-based scenario types.
        
        Distribution: Equal distribution (20% each)
        """
        print(f"\n🎲 Generating {num_scenarios} paper-based scenarios...")
        
        scenarios = []
        scenario_types = [
            'random_supplier_failure',
            'high_capacity_supplier_failure',
            'high_risk_manufacturer_failure',
            'central_distributor_failure',
            'multiple_transportation_delays'
        ]
        
        for i in tqdm(range(num_scenarios), desc="Simulating scenarios"):
            scenario_type = scenario_types[i % len(scenario_types)]
            
            if scenario_type == 'random_supplier_failure':
                scenario = self.scenario_1_random_supplier_failure(G)
            elif scenario_type == 'high_capacity_supplier_failure':
                scenario = self.scenario_2_high_capacity_supplier_failure(G)
            elif scenario_type == 'high_risk_manufacturer_failure':
                scenario = self.scenario_3_high_risk_manufacturer_failure(G)
            elif scenario_type == 'central_distributor_failure':
                scenario = self.scenario_4_central_distributor_failure(G)
            elif scenario_type == 'multiple_transportation_delays':
                scenario = self.scenario_5_multiple_transportation_delays(G)
            
            scenario['scenario_id'] = i
            scenarios.append(scenario)
        
        # Calculate statistics
        total_affected = sum(len(s['results']) for s in scenarios)
        avg_affected = total_affected / num_scenarios
        
        print(f"\n  ✓ Generated {num_scenarios} scenarios")
        print(f"  ✓ Average nodes affected per scenario: {avg_affected:.1f}")
        print(f"\n  📊 Scenario Distribution:")
        for st in scenario_types:
            count = sum(1 for s in scenarios if s['scenario_type'] == st)
            print(f"    {st}: {count} ({count/num_scenarios*100:.1f}%)")
        
        return scenarios
    
    def create_pyg_data_objects(self, G, node_df, edge_df, scenarios):
        """
        Create PyTorch Geometric Data objects.
        
        Node features (11 total):
        [0-5] capacity, cost_factor, risk_level, reliability, x, y
        [6-9] tier one-hot encoding
        [10] is_initially_disrupted (binary feature - added per scenario)
        
        Edge features (4 total):
        [0-3] lead_time, cost, capacity_share, disruption_prob
        """
        print("\n💾 Creating PyG Data objects...")
        
        # Prepare base node features (without is_initially_disrupted)
        base_features = torch.tensor(
            node_df[['capacity', 'cost_factor', 'risk_level', 'reliability', 'x', 'y']].values,
            dtype=torch.float
        )
        
        # Add tier one-hot encoding
        tier_encoding = torch.zeros((len(node_df), 4), dtype=torch.float)
        for idx, tier in enumerate(node_df['tier'].values):
            tier_encoding[idx, int(tier)] = 1.0
        
        base_features = torch.cat([base_features, tier_encoding], dim=1)
        
        print(f"  ✓ Base node features: {base_features.shape}")
        
        # Create edge_index
        edge_index = torch.tensor([
            edge_df['source'].values,
            edge_df['target'].values
        ], dtype=torch.long)
        
        # Create edge features
        num_edges = len(edge_df)
        base_edge_attr = torch.zeros((num_edges, 4), dtype=torch.float)
        
        # Feature 0: lead_time
        if 'weight' in edge_df.columns:
            weights = edge_df['weight'].values
            base_edge_attr[:, 0] = torch.tensor((weights - weights.mean()) / (weights.std() + 1e-8), dtype=torch.float)
        
        # Feature 1: cost
        for idx, row in edge_df.iterrows():
            source_cost = float(node_df.loc[row['source'], 'cost_factor'])
            target_cost = float(node_df.loc[row['target'], 'cost_factor'])
            base_edge_attr[idx, 1] = (source_cost + target_cost) / 2.0
        
        # Feature 2: capacity_share
        if 'weight' in edge_df.columns:
            base_edge_attr[:, 2] = torch.tensor(edge_df['weight'].values, dtype=torch.float)
        else:
            base_edge_attr[:, 2] = 1.0
        
        # Feature 3: disruption_prob
        for idx, row in edge_df.iterrows():
            source_risk = float(node_df.loc[row['source'], 'risk_level'])
            target_risk = float(node_df.loc[row['target'], 'risk_level'])
            base_edge_attr[idx, 3] = (source_risk + target_risk) / 2.0
        
        print(f"  ✓ Edge features: {base_edge_attr.shape}")
        
        num_nodes = len(node_df)
        data_objects = []
        
        for scenario in tqdm(scenarios, desc="Creating Data objects"):
            # Add is_initially_disrupted feature (binary: 0 or 1)
            is_disrupted = torch.zeros((num_nodes, 1), dtype=torch.float)
            
            # Mark initially disrupted nodes
            if 'disrupted_nodes' in scenario:
                for node_id in scenario['disrupted_nodes']:
                    is_disrupted[int(node_id), 0] = 1.0
            elif 'disrupted_edges' in scenario:
                # For edge disruptions, mark target nodes as disrupted
                for source, target in scenario['disrupted_edges']:
                    is_disrupted[int(target), 0] = 1.0
            
            # Concatenate base features with is_initially_disrupted
            x = torch.cat([base_features, is_disrupted], dim=1)
            
            y = torch.full((num_nodes,), -1, dtype=torch.long)
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            
            # Fill in labels
            for node_id, result in scenario['results'].items():
                node_id = int(node_id)
                y[node_id] = result['label']
                train_mask[node_id] = True
            
            edge_attr = base_edge_attr.clone()
            
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                train_mask=train_mask
            )
            
            data.scenario_id = scenario['scenario_id']
            data.scenario_type = scenario['scenario_type']
            
            # Store disrupted nodes/edges
            if 'disrupted_nodes' in scenario:
                data.disrupted_nodes = torch.tensor(scenario['disrupted_nodes'], dtype=torch.long)
            if 'disrupted_edges' in scenario:
                data.disrupted_edges = scenario['disrupted_edges']
            
            data_objects.append(data)
        
        print(f"  ✓ Created {len(data_objects)} Data objects")
        
        return data_objects
    
    def save_scenarios(self, data_objects, output_dir='scenario_graphs_paper'):
        """Save scenarios to disk."""
        print(f"\n💾 Saving scenarios to {output_dir}/...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual scenarios
        for data in tqdm(data_objects, desc="Saving"):
            filename = f"{output_dir}/scenario_{data.scenario_id:05d}.pt"
            torch.save(data, filename)
        
        # Save metadata
        metadata = {
            'num_scenarios': len(data_objects),
            'scenario_types': list(set(d.scenario_type for d in data_objects)),
            'node_features': 11,  # Updated: now includes is_initially_disrupted
            'edge_features': 4,
            'num_classes': 3
        }
        
        with open(f'{output_dir}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Saved {len(data_objects)} scenarios")
        print(f"  ✓ Saved metadata.json")


def load_preprocessed_graph(graph_file='supply_chain_graph.pt'):
    """Load preprocessed graph data."""
    print(f"\n📂 Loading preprocessed graph from {graph_file}...")
    
    data = torch.load(graph_file, weights_only=False)
    
    # Extract node data
    node_df = pd.DataFrame({
        'capacity': data.x[:, 0].numpy(),
        'cost_factor': data.x[:, 1].numpy(),
        'risk_level': data.x[:, 2].numpy(),
        'reliability': data.x[:, 3].numpy(),
        'x': data.x[:, 4].numpy(),
        'y': data.x[:, 5].numpy(),
        'tier': data.x[:, 6:10].argmax(dim=1).numpy(),
        'region': ['Region_' + str(int(i % 5)) for i in range(data.x.shape[0])]
    })
    
    # Extract edge data
    edge_df = pd.DataFrame({
        'source': data.edge_index[0].numpy(),
        'target': data.edge_index[1].numpy(),
        'weight': data.edge_attr[:, 2].numpy() if data.edge_attr is not None else [1.0] * data.edge_index.shape[1]
    })
    
    print(f"  ✓ Loaded {len(node_df)} nodes")
    print(f"  ✓ Loaded {len(edge_df)} edges")
    
    return node_df, edge_df


def main():
    """Main execution pipeline."""
    print("="*70)
    print("PAPER-BASED SCENARIO GENERATION")
    print("="*70)
    print("\nImplementing 5 scenarios from:")
    print("'Graph Neural Network-Based Predictive Modeling for")
    print(" Enhanced Supply Chain Resilience against Multi-Modal Disruptions'")
    
    # Load preprocessed data
    print("\n" + "="*70)
    print("STEP 1: LOADING PREPROCESSED DATA")
    print("="*70)
    
    node_df, edge_df = load_preprocessed_graph('supply_chain_graph.pt')
    
    # Initialize simulator
    simulator = PaperBasedDisruptionSimulator(seed=42)
    
    # Load graph
    print("\n" + "="*70)
    print("STEP 2: BUILDING GRAPH")
    print("="*70)
    
    G = simulator.load_graph(node_df, edge_df)
    
    # Generate scenarios
    print("\n" + "="*70)
    print("STEP 3: GENERATING PAPER-BASED SCENARIOS")
    print("="*70)
    
    num_scenarios = 1000
    scenarios = simulator.generate_paper_scenarios(G, num_scenarios=num_scenarios)
    
    # Create PyG Data objects
    print("\n" + "="*70)
    print("STEP 4: CREATING PYTORCH GEOMETRIC DATA")
    print("="*70)
    
    data_objects = simulator.create_pyg_data_objects(G, node_df, edge_df, scenarios)
    
    # Save scenarios
    print("\n" + "="*70)
    print("STEP 5: SAVING SCENARIOS")
    print("="*70)
    
    output_dir = 'scenario_graphs_paper'
    simulator.save_scenarios(data_objects, output_dir=output_dir)
    
    print("\n" + "="*70)
    print("✅ PAPER-BASED SCENARIO GENERATION COMPLETE!")
    print("="*70)
    
    print(f"\n📁 Output Directory: {output_dir}/")
    print(f"  ✓ {num_scenarios} scenario files")
    print(f"  ✓ 5 scenario types (20% each)")
    print(f"\n📊 Scenario Types:")
    print(f"  1. Random Supplier Failure (10 random suppliers)")
    print(f"  2. High-Capacity Supplier Failure (5 highest-capacity)")
    print(f"  3. High-Risk Manufacturer Failure (8 high-risk)")
    print(f"  4. Central Distributor Failure (4 highest betweenness)")
    print(f"  5. Multiple Transportation Delays (25 random edges)")
    print(f"\n⚙️  Node disruptions: Reliability reduced by 70%")
    print(f"⚙️  Edge disruptions: Lead time increased 2x")


if __name__ == "__main__":
    main()
