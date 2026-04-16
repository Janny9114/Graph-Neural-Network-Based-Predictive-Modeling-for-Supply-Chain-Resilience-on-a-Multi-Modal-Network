"""
Edge Disruption Scenario Generator
Extends generate_realistic_scenarios.py with edge-level disruptions.

New Scenario Types:
1. Transportation Route Disruption (weather, accidents)
2. Trade Restrictions (geopolitical conflicts)
3. Cyber Attacks on Logistics Network
4. Port/Shipping Lane Blockage
5. Infrastructure Failure (roads, rails)

Key Difference from Node Disruptions:
- Node disruptions: Nodes fail, cascades propagate through edges
- Edge disruptions: Edges fail/degrade, nodes fail due to supply bottlenecks
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
from generate_realistic_scenarios import RealisticDisruptionSimulator


class EdgeDisruptionSimulator(RealisticDisruptionSimulator):
    """
    Extends RealisticDisruptionSimulator with edge disruption capabilities.
    """
    
    def __init__(self, seed=42, edge_disruption_csv='simulation_disruption/global_supply_chain_disruption_edge_mapped.csv'):
        super().__init__(seed)
        self.edge_disruption_data = None
        self.edge_disruption_csv = edge_disruption_csv
        
    def load_edge_disruption_data(self):
        """Load edge disruption data from CSV."""
        print(f"\n📊 Loading edge disruption data from {self.edge_disruption_csv}...")
        
        try:
            df = pd.read_csv(self.edge_disruption_csv)
            print(f"  ✓ Loaded {len(df)} edge disruption records")
            
            # Analyze disruption types
            disruption_cols = [col for col in df.columns if col.startswith('Disruption_Event_')]
            print(f"\n  📊 Disruption Types:")
            for col in disruption_cols:
                count = df[col].sum()
                pct = count / len(df) * 100
                event_name = col.replace('Disruption_Event_', '')
                print(f"    - {event_name}: {count} ({pct:.1f}%)")
            
            self.edge_disruption_data = df
            return df
            
        except FileNotFoundError:
            print(f"  ⚠ Warning: {self.edge_disruption_csv} not found")
            print(f"  → Edge disruption scenarios will use synthetic data")
            return None
    
    def simulate_edge_cascade(self, G, base_buffers, disrupted_edges, edge_capacity_reduction):
        """
        Simulate cascade when edges are disrupted.
        
        NEW APPROACH: Keep edges in graph, mark them as disrupted in edge features
        
        Args:
            G: NetworkX graph
            base_buffers: Dict of base buffer values
            disrupted_edges: List of (source, target) tuples
            edge_capacity_reduction: Dict mapping edge to capacity reduction (0.0-1.0)
        
        Returns:
            Dict with results for each affected node AND edge disruption info
        """
        # Add stochastic variance to buffers
        buffers = {
            node: base_buffers[node] * np.random.uniform(0.8, 1.2)
            for node in G.nodes()
        }
        
        # DON'T modify graph - keep edges intact!
        # Instead, track disruption info for edge features
        edge_disruption_info = {}
        for edge, reduction in edge_capacity_reduction.items():
            if G.has_edge(edge[0], edge[1]):
                # Store disruption info for this edge
                edge_disruption_info[edge] = {
                    'is_disrupted': 1.0,
                    'disruption_severity': reduction,  # 0.0-1.0
                    'time_to_recovery': np.random.uniform(3, 30)  # 3-30 days
                }
        
        # Find all nodes affected by edge disruptions
        affected_nodes = set()
        for edge in disrupted_edges:
            affected_nodes.add(edge[1])  # Target nodes directly affected
        
        # PRIORITY 2 FIX: Use accumulator instead of visited set
        # This allows nodes to accumulate shortages from multiple sources
        from collections import defaultdict
        shortage_accumulator = defaultdict(float)
        processed = set()
        queue = deque([(node, 0.0, 0) for node in affected_nodes])  # (node, shortage, depth)
        
        # Track max time_to_recovery for each node
        time_to_recovery_map = {}
        for edge, info in edge_disruption_info.items():
            target = edge[1]
            if target not in time_to_recovery_map:
                time_to_recovery_map[target] = info['time_to_recovery']
            else:
                time_to_recovery_map[target] = max(time_to_recovery_map[target], info['time_to_recovery'])
        
        # Calculate supply shortage for each affected node
        results = {}
        
        while queue:
            current_node, accumulated_shortage, depth = queue.popleft()
            
            # Calculate supply shortage due to edge disruptions
            total_inflow = 0
            reduced_inflow = 0
            
            for predecessor in G.predecessors(current_node):
                edge = (predecessor, current_node)
                original_weight = G[predecessor][current_node].get('weight', 1.0)
                
                if edge in edge_capacity_reduction:
                    reduction = edge_capacity_reduction[edge]
                    reduced_weight = original_weight * (1 - reduction)
                    total_inflow += original_weight
                    reduced_inflow += reduced_weight
                else:
                    total_inflow += original_weight
                    reduced_inflow += original_weight
            
            # Calculate shortage as percentage
            if total_inflow > 0:
                shortage_pct = (total_inflow - reduced_inflow) / total_inflow
            else:
                shortage_pct = 0.0
            
            # Add accumulated shortage from upstream
            total_shortage = min(1.0, shortage_pct + accumulated_shortage)
            
            # Accumulate shortage for this node
            shortage_accumulator[current_node] += total_shortage
            
            # Process node only once all sources have contributed
            if current_node not in processed:
                processed.add(current_node)
                
                # Get node properties
                capacity = G.nodes[current_node]['capacity']
                buffer = buffers[current_node]
                
                # PRIORITY 3: Add time dynamics
                # Impact = shortage_rate * time_to_recovery - buffer
                time_to_recovery = time_to_recovery_map.get(current_node, 7.0)  # Default 7 days
                time_factor = time_to_recovery / 30.0  # Normalize to 0-1
                
                # Calculate impact with time dynamics
                impact_units = capacity * shortage_accumulator[current_node] * time_factor
                
                # Determine if node survives
                remaining_impact = max(0, impact_units - buffer)
                
                # 4-CLASS LABELS: More realistic severity modeling
                # Calculate impact ratio (how much of the shortage remains after buffer)
                if impact_units > 0:
                    impact_ratio = remaining_impact / impact_units
                else:
                    impact_ratio = 0.0
                
                # 3-CLASS LABELS: Balanced classification
                # 0 = Failed (>60% impact remains - severe disruption)
                # 1 = Degraded (<60% impact remains - partial disruption)
                # 2 = Normal (0% impact - fully operational)
                if remaining_impact == 0:
                    label = 2  # Normal (fully operational)
                elif impact_ratio < 0.6:
                    label = 1  # Degraded (some impact, but manageable)
                else:
                    label = 0  # Failed (severe impact)
                
                # Store results
                results[current_node] = {
                    'buffer': buffer,
                    'production_impact_pct': shortage_accumulator[current_node],
                    'production_impact_units': impact_units,
                    'remaining_impact': remaining_impact,
                    'label': label,
                    'is_disrupted': 1 if current_node in affected_nodes else 0,
                    'time_to_recovery': time_to_recovery
                }
                
                # If node failed, propagate shortage downstream
                if remaining_impact > 0:
                    downstream = list(G.successors(current_node))
                    
                    if downstream:
                        # Calculate propagated shortage
                        propagated_shortage = remaining_impact / capacity
                        
                        for neighbor in downstream:
                            queue.append((neighbor, propagated_shortage * 0.8, depth + 1))  # 80% propagation
                            # Update time_to_recovery for downstream nodes
                            if neighbor not in time_to_recovery_map:
                                time_to_recovery_map[neighbor] = time_to_recovery + 3  # Add 3 days delay
        
        # ✅ Add label = -1 for unaffected nodes
        # Unaffected = nodes not in processed set (never touched by cascade)
        for node in G.nodes():
            if node not in processed:
                results[node] = {
                    'buffer': buffers[node],
                    'production_impact_pct': 0.0,
                    'production_impact_units': 0.0,
                    'remaining_impact': 0.0,
                    'label': -1,  # Unaffected (exclude from training)
                    'is_disrupted': 0,
                    'time_to_recovery': 0.0
                }
        
        return results
    
    def simulate_transportation_route_disruption(self, G, base_buffers):
        """
        Scenario 1: Transportation route disruption (weather, accidents).
        Affects 1-5 random edges with varying severity.
        """
        edges = list(G.edges())
        
        # Select 1-5 random edges
        num_disrupted = np.random.randint(1, 6)
        disrupted_edges = [edges[i] for i in np.random.choice(len(edges), size=num_disrupted, replace=False)]
        
        # Assign severity based on weather/accident type
        severity_type = np.random.choice(['minor', 'moderate', 'severe'], p=[0.5, 0.35, 0.15])
        
        edge_capacity_reduction = {}
        for edge in disrupted_edges:
            if severity_type == 'minor':
                reduction = np.random.uniform(0.2, 0.4)  # 20-40% capacity loss
            elif severity_type == 'moderate':
                reduction = np.random.uniform(0.4, 0.7)  # 40-70% capacity loss
            else:  # severe
                reduction = np.random.uniform(0.7, 1.0)  # 70-100% capacity loss
            
            edge_capacity_reduction[edge] = reduction
        
        # Simulate cascade
        results = self.simulate_edge_cascade(G, base_buffers, disrupted_edges, edge_capacity_reduction)
        
        return {
            'scenario_type': 'transportation_route_disruption',
            'disrupted_edges': disrupted_edges,
            'edge_capacity_reduction': edge_capacity_reduction,
            'severity_type': severity_type,
            'results': results
        }
    
    def simulate_trade_restriction(self, G, base_buffers):
        """
        Scenario 2: Trade restrictions (geopolitical conflicts).
        Affects cross-border edges between two regions.
        """
        # Get unique regions
        regions = list(set(G.nodes[n]['region'] for n in G.nodes()))
        
        if len(regions) < 2:
            # Fallback: random edge disruption
            return self.simulate_transportation_route_disruption(G, base_buffers)
        
        # Select two regions in conflict
        region1, region2 = np.random.choice(regions, size=2, replace=False)
        
        # Find cross-border edges
        cross_border_edges = []
        for u, v in G.edges():
            if (G.nodes[u]['region'] == region1 and G.nodes[v]['region'] == region2) or \
               (G.nodes[u]['region'] == region2 and G.nodes[v]['region'] == region1):
                cross_border_edges.append((u, v))
        
        if len(cross_border_edges) == 0:
            # Fallback: random edge disruption
            return self.simulate_transportation_route_disruption(G, base_buffers)
        
        # Tariffs/restrictions reduce capacity by 30-70%
        edge_capacity_reduction = {}
        for edge in cross_border_edges:
            reduction = np.random.uniform(0.3, 0.7)
            edge_capacity_reduction[edge] = reduction
        
        # Simulate cascade
        results = self.simulate_edge_cascade(G, base_buffers, cross_border_edges, edge_capacity_reduction)
        
        return {
            'scenario_type': 'trade_restriction',
            'disrupted_edges': cross_border_edges,
            'edge_capacity_reduction': edge_capacity_reduction,
            'affected_regions': [region1, region2],
            'results': results
        }
    
    def simulate_cyber_attack_logistics(self, G, base_buffers):
        """
        Scenario 3: Cyber attack on logistics network.
        Affects 10-30% of random edges (communication/coordination disruption).
        """
        edges = list(G.edges())
        
        # Cyber attack affects 10-30% of edges
        attack_severity = np.random.choice(['targeted', 'widespread'], p=[0.6, 0.4])
        
        if attack_severity == 'targeted':
            # Target high-betweenness edges (critical routes)
            edge_betweenness = nx.edge_betweenness_centrality(G)
            sorted_edges = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)
            num_disrupted = int(len(edges) * np.random.uniform(0.1, 0.2))  # 10-20%
            disrupted_edges = [edge for edge, _ in sorted_edges[:num_disrupted]]
        else:
            # Widespread random attack
            num_disrupted = int(len(edges) * np.random.uniform(0.2, 0.3))  # 20-30%
            disrupted_edges = [edges[i] for i in np.random.choice(len(edges), size=num_disrupted, replace=False)]
        
        # Cyber attack causes 50-90% capacity reduction (coordination failure)
        edge_capacity_reduction = {}
        for edge in disrupted_edges:
            reduction = np.random.uniform(0.5, 0.9)
            edge_capacity_reduction[edge] = reduction
        
        # Simulate cascade
        results = self.simulate_edge_cascade(G, base_buffers, disrupted_edges, edge_capacity_reduction)
        
        return {
            'scenario_type': 'cyber_attack_logistics',
            'disrupted_edges': disrupted_edges,
            'edge_capacity_reduction': edge_capacity_reduction,
            'attack_severity': attack_severity,
            'results': results
        }
    
    def simulate_multi_hop_disruption(self, G, base_buffers):
        """
        NEW Scenario: Multi-hop supply chain disruption.
        Disrupts nodes at multiple tiers simultaneously to test deep cascade effects.
        
        Strategy:
        - Disrupt 2-4 suppliers (tier 0)
        - Disrupt 1-2 manufacturers (tier 1)
        - Disrupt 5-10 critical edges connecting them
        
        This creates a complex, multi-level disruption that tests GNN's ability
        to reason about deep cascades through the supply chain.
        """
        # Find nodes by tier
        tier_0 = [n for n in G.nodes() if G.nodes[n]['tier'] == 0]  # Suppliers
        tier_1 = [n for n in G.nodes() if G.nodes[n]['tier'] == 1]  # Manufacturers
        tier_2 = [n for n in G.nodes() if G.nodes[n]['tier'] == 2]  # Distributors
        
        # Fallback if tiers not available
        if len(tier_0) == 0:
            tier_0 = list(G.nodes())[:50]
        if len(tier_1) == 0:
            tier_1 = list(G.nodes())[50:100]
        if len(tier_2) == 0:
            tier_2 = list(G.nodes())[100:150]
        
        # Select disrupted nodes at multiple tiers
        num_tier_0 = np.random.randint(2, 5)  # 2-4 suppliers
        num_tier_1 = np.random.randint(1, 3)  # 1-2 manufacturers
        
        disrupted_tier_0 = np.random.choice(tier_0, size=min(num_tier_0, len(tier_0)), replace=False)
        disrupted_tier_1 = np.random.choice(tier_1, size=min(num_tier_1, len(tier_1)), replace=False)
        
        all_disrupted_nodes = list(disrupted_tier_0) + list(disrupted_tier_1)
        
        # Initial production impacts (60-95%)
        initial_impact_pcts = [np.random.uniform(0.6, 0.95) for _ in all_disrupted_nodes]
        
        # Also disrupt critical edges connecting these tiers
        disrupted_edges = []
        edge_capacity_reduction = {}
        
        # Find edges connecting disrupted nodes to downstream
        for node in all_disrupted_nodes:
            for successor in G.successors(node):
                edge = (node, successor)
                if edge not in disrupted_edges:
                    # 50% chance to also disrupt the edge
                    if np.random.random() < 0.5:
                        disrupted_edges.append(edge)
                        edge_capacity_reduction[edge] = np.random.uniform(0.5, 0.8)
        
        # Add some random critical edges (high betweenness)
        if len(list(G.edges())) > 0:
            edge_betweenness = nx.edge_betweenness_centrality(G)
            sorted_edges = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)
            num_additional = np.random.randint(3, 8)  # 3-7 additional critical edges
            
            for edge, _ in sorted_edges[:num_additional]:
                if edge not in disrupted_edges:
                    disrupted_edges.append(edge)
                    edge_capacity_reduction[edge] = np.random.uniform(0.4, 0.7)
        
        # Simulate node cascade first
        node_results = self.simulate_multi_node_cascade(G, base_buffers, all_disrupted_nodes, initial_impact_pcts)
        
        # Then simulate edge cascade on top
        edge_results = self.simulate_edge_cascade(G, base_buffers, disrupted_edges, edge_capacity_reduction)
        
        # Merge results (take worst case for each node)
        combined_results = {}
        all_affected_nodes = set(node_results.keys()) | set(edge_results.keys())
        
        for node in all_affected_nodes:
            node_label = node_results.get(node, {}).get('label', 2)
            edge_label = edge_results.get(node, {}).get('label', 2)
            
            # Take worst case (lower label = worse)
            combined_label = min(node_label, edge_label)
            
            # Combine other metrics
            combined_results[node] = {
                'label': combined_label,
                'is_disrupted': 1 if node in all_disrupted_nodes else 0,
                'node_impact': node_results.get(node, {}),
                'edge_impact': edge_results.get(node, {})
            }
        
        return {
            'scenario_type': 'multi_hop_disruption',
            'disrupted_nodes': all_disrupted_nodes,
            'disrupted_edges': disrupted_edges,
            'edge_capacity_reduction': edge_capacity_reduction,
            'initial_impact_pcts': initial_impact_pcts,
            'num_tiers_affected': 2,
            'results': combined_results
        }
    
    def simulate_major_supplier_failure(self, G, base_buffers):
        """
        Scenario 4: Major supplier failure.
        Disrupts 1-3 suppliers with highest capacity, representing critical supplier disruption.
        """
        # Find supplier nodes (tier 0)
        supplier_nodes = [n for n in G.nodes() if G.nodes[n]['tier'] == 0]
        
        if len(supplier_nodes) == 0:
            # Fallback: use high-capacity nodes
            supplier_nodes = list(G.nodes())
        
        # Sort by capacity
        suppliers_by_capacity = sorted(
            [(n, G.nodes[n]['capacity']) for n in supplier_nodes],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Select 1-3 top suppliers
        num_suppliers = np.random.randint(1, 4)
        failed_suppliers = [n for n, _ in suppliers_by_capacity[:min(num_suppliers, len(suppliers_by_capacity))]]
        
        # Find all outgoing edges from failed suppliers
        disrupted_edges = []
        for supplier in failed_suppliers:
            for successor in G.successors(supplier):
                disrupted_edges.append((supplier, successor))
        
        # Failure severity
        failure_type = np.random.choice(['partial', 'complete'], p=[0.6, 0.4])
        
        edge_capacity_reduction = {}
        for edge in disrupted_edges:
            if failure_type == 'partial':
                reduction = np.random.uniform(0.6, 0.85)  # 60-85% reduction
            else:  # complete
                reduction = np.random.uniform(0.9, 1.0)  # 90-100% reduction
            
            edge_capacity_reduction[edge] = reduction
        
        # Simulate cascade
        results = self.simulate_edge_cascade(G, base_buffers, disrupted_edges, edge_capacity_reduction)
        
        return {
            'scenario_type': 'major_supplier_failure',
            'disrupted_edges': disrupted_edges,
            'edge_capacity_reduction': edge_capacity_reduction,
            'failed_suppliers': failed_suppliers,
            'failure_type': failure_type,
            'results': results
        }
    
    def simulate_infrastructure_failure(self, G, base_buffers):
        """
        Scenario 5: Infrastructure failure (roads, rails).
        Affects edges in a geographic region (spatial clustering).
        """
        nodes = list(G.nodes())
        
        # Select epicenter
        epicenter = np.random.choice(nodes)
        epicenter_x = G.nodes[epicenter]['x']
        epicenter_y = G.nodes[epicenter]['y']
        
        # Infrastructure failure radius (in Z-score units)
        failure_type = np.random.choice(['local', 'regional'], p=[0.6, 0.4])
        
        if failure_type == 'local':
            radius = np.random.uniform(0.3, 0.8)  # Local infrastructure
        else:
            radius = np.random.uniform(0.8, 1.5)  # Regional infrastructure
        
        # Find edges within radius
        disrupted_edges = []
        for u, v in G.edges():
            # Check if either endpoint is within radius
            u_x, u_y = G.nodes[u]['x'], G.nodes[u]['y']
            v_x, v_y = G.nodes[v]['x'], G.nodes[v]['y']
            
            dist_u = np.sqrt((u_x - epicenter_x)**2 + (u_y - epicenter_y)**2)
            dist_v = np.sqrt((v_x - epicenter_x)**2 + (v_y - epicenter_y)**2)
            
            if dist_u <= radius or dist_v <= radius:
                disrupted_edges.append((u, v))
        
        if len(disrupted_edges) == 0:
            # Fallback: expand radius
            radius *= 1.5
            for u, v in G.edges():
                u_x, u_y = G.nodes[u]['x'], G.nodes[u]['y']
                v_x, v_y = G.nodes[v]['x'], G.nodes[v]['y']
                
                dist_u = np.sqrt((u_x - epicenter_x)**2 + (u_y - epicenter_y)**2)
                dist_v = np.sqrt((v_x - epicenter_x)**2 + (v_y - epicenter_y)**2)
                
                if dist_u <= radius or dist_v <= radius:
                    disrupted_edges.append((u, v))
        
        # Infrastructure failure: 60-95% capacity reduction
        edge_capacity_reduction = {}
        for edge in disrupted_edges:
            reduction = np.random.uniform(0.6, 0.95)
            edge_capacity_reduction[edge] = reduction
        
        # Simulate cascade
        results = self.simulate_edge_cascade(G, base_buffers, disrupted_edges, edge_capacity_reduction)
        
        return {
            'scenario_type': 'infrastructure_failure',
            'disrupted_edges': disrupted_edges,
            'edge_capacity_reduction': edge_capacity_reduction,
            'epicenter': epicenter,
            'radius': radius,
            'failure_type': failure_type,
            'results': results
        }
    
    def generate_scenarios_with_edge_disruptions(self, G, base_buffers, num_scenarios=10000):
        """
        Generate scenarios with BOTH node and edge disruptions.
        
        Distribution:
        - 40% Node-only disruptions (existing scenarios)
        - 40% Edge-only disruptions (new scenarios)
        - 20% Hybrid (node + edge disruptions)
        """
        print(f"\n🎲 Generating {num_scenarios} scenarios with edge disruptions...")
        
        scenarios = []
        
        # Define scenario types
        scenario_types = [
            # Node-only (40%)
            'regional_failure_variable',
            'distributor_hub_failure',
            'random_supplier_failure',
            'central',
            # Edge-only (40%)
            'transportation_route_disruption',
            'trade_restriction',
            'cyber_attack_logistics',
            'major_supplier_failure',
            # Hybrid (20%)
            'infrastructure_failure',
            'hybrid_node_edge'
        ]
        
        for i in tqdm(range(num_scenarios), desc="Simulating scenarios"):
            scenario_type = scenario_types[i % len(scenario_types)]
            
            # Node-only scenarios (use parent class methods)
            if scenario_type == 'regional_failure_variable':
                scenario = self.simulate_regional_failure_variable(G, base_buffers)
                scenario['scenario_id'] = i
                scenario['disruption_category'] = 'node_only'
                scenarios.append(scenario)
                
            elif scenario_type == 'distributor_hub_failure':
                # Find distributor nodes (tier 2) or use betweenness centrality
                distributor_nodes = [n for n in G.nodes() if G.nodes[n]['tier'] == 2]
                
                if len(distributor_nodes) == 0:
                    # Fallback: use all nodes
                    distributor_nodes = list(G.nodes())
                
                # Calculate betweenness centrality
                betweenness = nx.betweenness_centrality(G)
                
                # Filter to distributors only and sort by betweenness
                distributor_betweenness = [(n, betweenness[n]) for n in distributor_nodes]
                distributor_betweenness.sort(key=lambda x: x[1], reverse=True)
                
                # Select top 10% by betweenness
                top_10_pct = max(4, len(distributor_betweenness) // 10)
                hub_candidates = [n for n, _ in distributor_betweenness[:top_10_pct]]
                
                # Select 2-4 random hubs from top 10%
                num_hubs = np.random.randint(2, 5)
                initial_nodes = list(np.random.choice(hub_candidates, size=min(num_hubs, len(hub_candidates)), replace=False))
                initial_impact_pcts = [np.random.uniform(0.1, 1.0) for _ in range(len(initial_nodes))]
                results = self.simulate_multi_node_cascade(G, base_buffers, initial_nodes, initial_impact_pcts)
                
                scenarios.append({
                    'scenario_id': i,
                    'scenario_type': scenario_type,
                    'disruption_category': 'node_only',
                    'initial_node': initial_nodes,
                    'initial_impact_pct': initial_impact_pcts,
                    'results': results
                })
                
            elif scenario_type == 'random_supplier_failure':
                # Find supplier nodes (tier 0)
                supplier_nodes = [n for n in G.nodes() if G.nodes[n]['tier'] == 0]
                
                if len(supplier_nodes) < 10:
                    # Fallback: use all nodes if not enough suppliers
                    supplier_nodes = list(G.nodes())
                
                # Select 10 random suppliers
                num_suppliers = min(10, len(supplier_nodes))
                initial_nodes = list(np.random.choice(supplier_nodes, size=num_suppliers, replace=False))
                initial_impact_pcts = [np.random.uniform(0.1, 1.0) for _ in range(num_suppliers)]
                results = self.simulate_multi_node_cascade(G, base_buffers, initial_nodes, initial_impact_pcts)
                
                scenarios.append({
                    'scenario_id': i,
                    'scenario_type': scenario_type,
                    'disruption_category': 'node_only',
                    'initial_node': initial_nodes,
                    'initial_impact_pct': initial_impact_pcts,
                    'results': results
                })
                
            elif scenario_type == 'central':
                # Select 5 nodes from top 10% most connected nodes
                degree_centrality = nx.degree_centrality(G)
                sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
                top_10_pct = max(5, len(sorted_nodes) // 10)
                central_candidates = [n for n, _ in sorted_nodes[:top_10_pct]]
                
                # Select 5 random nodes from top 10%
                initial_nodes = list(np.random.choice(central_candidates, size=min(5, len(central_candidates)), replace=False))
                initial_impact_pcts = [np.random.uniform(0.1, 1.0) for _ in range(len(initial_nodes))]
                results = self.simulate_multi_node_cascade(G, base_buffers, initial_nodes, initial_impact_pcts)
                
                scenarios.append({
                    'scenario_id': i,
                    'scenario_type': scenario_type,
                    'disruption_category': 'node_only',
                    'initial_node': initial_nodes,
                    'initial_impact_pct': initial_impact_pcts,
                    'results': results
                })
            
            # Edge-only scenarios
            elif scenario_type == 'transportation_route_disruption':
                scenario = self.simulate_transportation_route_disruption(G, base_buffers)
                scenario['scenario_id'] = i
                scenario['disruption_category'] = 'edge_only'
                scenarios.append(scenario)
                
            elif scenario_type == 'trade_restriction':
                scenario = self.simulate_trade_restriction(G, base_buffers)
                scenario['scenario_id'] = i
                scenario['disruption_category'] = 'edge_only'
                scenarios.append(scenario)
                
            elif scenario_type == 'cyber_attack_logistics':
                scenario = self.simulate_cyber_attack_logistics(G, base_buffers)
                scenario['scenario_id'] = i
                scenario['disruption_category'] = 'edge_only'
                scenarios.append(scenario)
                
            elif scenario_type == 'major_supplier_failure':
                scenario = self.simulate_major_supplier_failure(G, base_buffers)
                scenario['scenario_id'] = i
                scenario['disruption_category'] = 'edge_only'
                scenarios.append(scenario)
            
            # Hybrid scenarios
            elif scenario_type == 'infrastructure_failure':
                scenario = self.simulate_infrastructure_failure(G, base_buffers)
                scenario['scenario_id'] = i
                scenario['disruption_category'] = 'hybrid'
                scenarios.append(scenario)
                
            elif scenario_type == 'hybrid_node_edge':
                # Combine regional node failure + transportation disruption
                node_scenario = self.simulate_regional_failure_variable(G, base_buffers)
                edge_scenario = self.simulate_transportation_route_disruption(G, base_buffers)
                
                # Merge results (nodes affected by both)
                merged_results = node_scenario['results'].copy()
                for node_id, edge_result in edge_scenario['results'].items():
                    if node_id in merged_results:
                        # Accumulate impacts
                        merged_results[node_id]['production_impact_pct'] += edge_result['production_impact_pct']
                        merged_results[node_id]['production_impact_pct'] = min(1.0, merged_results[node_id]['production_impact_pct'])
                        # Recalculate label
                        capacity = G.nodes[node_id]['capacity']
                        impact_units = capacity * merged_results[node_id]['production_impact_pct']
                        remaining = max(0, impact_units - merged_results[node_id]['buffer'])
                        merged_results[node_id]['label'] = 1 if remaining == 0 else 0
                    else:
                        merged_results[node_id] = edge_result
                
                scenarios.append({
                    'scenario_id': i,
                    'scenario_type': 'hybrid_node_edge',
                    'disruption_category': 'hybrid',
                    'node_disruption': node_scenario,
                    'edge_disruption': edge_scenario,
                    'results': merged_results
                })
        
        # Calculate statistics
        total_affected = sum(len(s['results']) for s in scenarios)
        avg_affected = total_affected / num_scenarios
        
        # Count categories
        node_only = sum(1 for s in scenarios if s.get('disruption_category') == 'node_only')
        edge_only = sum(1 for s in scenarios if s.get('disruption_category') == 'edge_only')
        hybrid = sum(1 for s in scenarios if s.get('disruption_category') == 'hybrid')
        
        print(f"\n  ✓ Generated {num_scenarios} scenarios")
        print(f"  ✓ Average nodes affected per scenario: {avg_affected:.1f}")
        print(f"\n  📊 Disruption Categories:")
        print(f"    Node-only: {node_only} ({node_only/num_scenarios*100:.1f}%)")
        print(f"    Edge-only: {edge_only} ({edge_only/num_scenarios*100:.1f}%)")
        print(f"    Hybrid: {hybrid} ({hybrid/num_scenarios*100:.1f}%)")
        
        return scenarios
    
    def create_pyg_data_objects(self, G, node_df, edge_df, scenarios):
        """
        Override parent method to add FAIR features for GNN vs ML comparison.
        
        Node features (10 total):
        [0] capacity
        [1] cost_factor
        [2] risk_level
        [3] reliability
        [4] x (location)
        [5] y (location)
        [6] tier_supplier (one-hot)
        [7] tier_manufacturer (one-hot)
        [8] tier_distributor (one-hot)
        [9] tier_retailer (one-hot)
        
        Edge features (4 total - NO DISRUPTION INFO):
        [0] lead_time (normalized) - STATIC
        [1] cost (normalized) - STATIC
        [2] capacity_share (weight) - STATIC
        [3] disruption_prob (risk-based) - STATIC
        """
        print("\n💾 Creating PyG Data objects with FAIR features (tier one-hot, no buffer, no disruption signals)...")
        
        # Prepare base features (first 6 features including x,y)
        base_features = torch.tensor(
            node_df[['capacity', 'cost_factor', 'risk_level', 'reliability', 'x', 'y']].values,
            dtype=torch.float
        )
        
        # Add tier one-hot encoding (4 dimensions)
        tier_encoding = torch.zeros((len(node_df), 4), dtype=torch.float)
        for idx, tier in enumerate(node_df['tier'].values):
            tier_encoding[idx, int(tier)] = 1.0
        
        # Concatenate base features + tier encoding
        base_features = torch.cat([base_features, tier_encoding], dim=1)
        
        print(f"  ✓ Node features: {base_features.shape} (6 base + 4 tier one-hot)")
        
        # Create edge_index
        edge_index = torch.tensor([
            edge_df['source'].values,
            edge_df['target'].values
        ], dtype=torch.long)
        
        # Create BASE edge features (4 features - NO DISRUPTION INFO)
        num_edges = len(edge_df)
        base_edge_attr = torch.zeros((num_edges, 4), dtype=torch.float)
        
        # Feature 0: lead_time (based on edge weight, normalized)
        if 'weight' in edge_df.columns:
            weights = edge_df['weight'].values
            base_edge_attr[:, 0] = torch.tensor((weights - weights.mean()) / (weights.std() + 1e-8), dtype=torch.float)
        
        # Feature 1: cost (based on source/target cost factors)
        for idx, row in edge_df.iterrows():
            source_cost = float(node_df.loc[row['source'], 'cost_factor'])
            target_cost = float(node_df.loc[row['target'], 'cost_factor'])
            base_edge_attr[idx, 1] = (source_cost + target_cost) / 2.0
        
        # Feature 2: capacity_share (weight)
        if 'weight' in edge_df.columns:
            base_edge_attr[:, 2] = torch.tensor(edge_df['weight'].values, dtype=torch.float)
        else:
            base_edge_attr[:, 2] = 1.0
        
        # Feature 3: disruption_prob (based on source/target risk)
        for idx, row in edge_df.iterrows():
            source_risk = float(node_df.loc[row['source'], 'risk_level'])
            target_risk = float(node_df.loc[row['target'], 'risk_level'])
            base_edge_attr[idx, 3] = (source_risk + target_risk) / 2.0
        
        print(f"  ✓ Created base edge features: {base_edge_attr.shape} (4 features, NO disruption signals)")
        
        num_nodes = len(node_df)
        data_objects = []
        
        for scenario in tqdm(scenarios, desc="Creating Data objects"):
            # CRITICAL FIX: Add is_initially_disrupted feature (binary: 0 or 1)
            is_disrupted = torch.zeros((num_nodes, 1), dtype=torch.float)
            
            # Mark initially disrupted nodes
            if 'initial_node' in scenario and scenario['initial_node'] is not None:
                if isinstance(scenario['initial_node'], list):
                    for node_id in scenario['initial_node']:
                        is_disrupted[int(node_id), 0] = 1.0
                else:
                    is_disrupted[int(scenario['initial_node']), 0] = 1.0
            
            # For edge disruptions, mark target nodes as disrupted
            if 'disrupted_edges' in scenario and scenario['disrupted_edges'] is not None:
                for edge in scenario['disrupted_edges']:
                    if isinstance(edge, tuple) and len(edge) >= 2:
                        target = edge[1]
                        is_disrupted[int(target), 0] = 1.0
            
            # Concatenate base features with is_initially_disrupted
            x = torch.cat([base_features, is_disrupted], dim=1)
            
            # Initialize labels and mask
            y = torch.full((num_nodes,), -1, dtype=torch.long)
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            
            # Fill in scenario-specific node data
            for node_id, result in scenario['results'].items():
                node_id = int(node_id)
                y[node_id] = result['label']
                train_mask[node_id] = True
            
            # Use STATIC edge features only
            edge_attr = base_edge_attr.clone()
            
            # Create Data object with FAIR features
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,  # ✅ Static edge features only!
                y=y,
                train_mask=train_mask
            )
            
            # Add metadata
            data.scenario_id = scenario['scenario_id']
            data.scenario_type = scenario['scenario_type']
            data.disruption_category = scenario.get('disruption_category', 'unknown')
            
            # Handle initial_node
            if 'initial_node' in scenario and scenario['initial_node'] is not None:
                if isinstance(scenario['initial_node'], list):
                    data.initial_nodes = torch.tensor(scenario['initial_node'], dtype=torch.long)
                    data.initial_impacts = torch.tensor(scenario['initial_impact_pct'], dtype=torch.float)
                    data.num_initial_disruptions = len(scenario['initial_node'])
                else:
                    data.initial_nodes = torch.tensor([scenario['initial_node']], dtype=torch.long)
                    data.initial_impacts = torch.tensor([scenario['initial_impact_pct']], dtype=torch.float)
                    data.num_initial_disruptions = 1
            else:
                data.initial_nodes = torch.tensor([], dtype=torch.long)
                data.initial_impacts = torch.tensor([], dtype=torch.float)
                data.num_initial_disruptions = 0
            
            # Handle disrupted_edges
            if 'disrupted_edges' in scenario and scenario['disrupted_edges'] is not None:
                data.disrupted_edges = scenario['disrupted_edges']
                data.num_edge_disruptions = len(scenario['disrupted_edges'])
            else:
                data.disrupted_edges = []
                data.num_edge_disruptions = 0
            
            data_objects.append(data)
        
        print(f"  ✓ Created {len(data_objects)} Data objects")
        print(f"  ✓ Node feature dimensions: {data_objects[0].x.shape} (6 base + 4 tier + 1 disruption indicator)")
        print(f"  ✓ Edge feature dimensions: {data_objects[0].edge_attr.shape} (4 static features)")
        print(f"  ✓ Node features: [capacity, cost_factor, risk_level, reliability, x, y, tier_supplier, tier_manufacturer, tier_distributor, tier_retailer, is_initially_disrupted]")
        print(f"  ✓ Edge features: [lead_time, cost, capacity_share, disruption_prob]")
        print(f"  ✓ Models can now see WHERE disruptions originated!")
        
        return data_objects


def main():
    """Main execution pipeline with edge disruptions."""
    print("="*70)
    print("EDGE DISRUPTION SCENARIO GENERATION")
    print("="*70)
    
    # Load preprocessed data
    from generate_realistic_scenarios import load_preprocessed_graph
    
    print("\n" + "="*70)
    print("STEP 1: LOADING PREPROCESSED DATA")
    print("="*70)
    
    node_df, edge_df = load_preprocessed_graph('supply_chain_graph.pt')
    
    # Initialize simulator
    simulator = EdgeDisruptionSimulator(seed=42)
    
    # Load edge disruption data
    simulator.load_edge_disruption_data()
    
    # Load graph
    print("\n" + "="*70)
    print("STEP 2: BUILDING GRAPH")
    print("="*70)
    
    G = simulator.load_graph(node_df, edge_df)
    
    # Store graph in simulator for calculate_base_buffers to use
    simulator.G = G
    
    # Calculate base buffers
    print("\n" + "="*70)
    print("STEP 3: CALCULATING BASE BUFFERS")
    print("="*70)
    
    base_buffers = simulator.calculate_base_buffers()
    print(f"  ✓ Calculated buffers for {len(base_buffers)} nodes")
    
    # Generate scenarios with edge disruptions
    print("\n" + "="*70)
    print("STEP 4: GENERATING SCENARIOS WITH EDGE DISRUPTIONS")
    print("="*70)
    
    num_scenarios = 1000
    scenarios = simulator.generate_scenarios_with_edge_disruptions(G, base_buffers, num_scenarios=num_scenarios)
    
    # Create PyG Data objects
    print("\n" + "="*70)
    print("STEP 5: CREATING PYTORCH GEOMETRIC DATA")
    print("="*70)
    
    data_objects = simulator.create_pyg_data_objects(G, node_df, edge_df, scenarios)
    
    # Save scenarios
    print("\n" + "="*70)
    print("STEP 6: SAVING SCENARIOS")
    print("="*70)
    
    output_dir = 'scenario_graphs_edge_disruptions'
    simulator.save_scenarios(data_objects, output_dir=output_dir)
    
    print("\n" + "="*70)
    print("✅ EDGE DISRUPTION SCENARIO GENERATION COMPLETE!")
    print("="*70)
    
    print(f"\n📁 Output Directory: {output_dir}/")
    print(f"  ✓ {num_scenarios} scenario files")
    print(f"  ✓ Includes node-only, edge-only, and hybrid disruptions")
    
    print("\n🎯 Expected GNN Advantage: +15-25% (up from +3.5%)")
    print("  → Edge disruptions require graph structure understanding")
    print("  → ML can't see edges, will struggle significantly")


if __name__ == "__main__":
    main()
