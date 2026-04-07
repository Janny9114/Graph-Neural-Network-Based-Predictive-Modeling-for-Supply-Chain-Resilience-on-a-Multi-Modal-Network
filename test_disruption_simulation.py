"""
Enhanced Disruption Simulation with Cascading Propagation
Implements graph-based disruption propagation as described in research papers
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Set
import torch

class CascadingDisruptionSimulator:
    """
    Implements cascading disruption propagation through supply chain network.
    
    Key improvements over baseline:
    1. Disruptions propagate through edges (2-3 hops)
    2. Impact depends on edge weights (dependency strength)
    3. Network centrality affects resilience
    4. Downstream nodes affected by upstream failures
    """
    
    def __init__(
        self,
        historical_data_path: str = "external_disruption_data/supply_chain_disruption_recovery.csv",
        resilience_threshold: float = 0.6,
        propagation_decay: float = 0.7,
        max_hops: int = 3,
        seed: int = 42
    ):
        """
        Initialize cascading disruption simulator.
        
        Args:
            historical_data_path: Path to historical disruption data
            resilience_threshold: Threshold for binary classification
            propagation_decay: Impact decay factor per hop (0.7 = 30% reduction)
            max_hops: Maximum propagation distance
            seed: Random seed
        """
        self.historical_data = pd.read_csv(historical_data_path)
        self.resilience_threshold = resilience_threshold
        self.propagation_decay = propagation_decay
        self.max_hops = max_hops
        np.random.seed(seed)
        
        self._analyze_historical_patterns()
    
    def _analyze_historical_patterns(self):
        """Extract disruption patterns from historical data."""
        df = self.historical_data
        
        # Disruption probability by region
        self.region_disruption_prob = df.groupby('supplier_region').size() / len(df)
        
        # Disruption type distribution
        self.region_disruption_types = df.groupby(['supplier_region', 'disruption_type']).size().unstack(fill_value=0)
        self.region_disruption_types = self.region_disruption_types.div(
            self.region_disruption_types.sum(axis=1), axis=0
        )
        
        # Severity statistics
        self.disruption_severity_mean = df.groupby('disruption_type')['disruption_severity'].mean()
        self.disruption_severity_std = df.groupby('disruption_type')['disruption_severity'].std()
        
        print("Historical patterns analyzed for cascading simulation")
    
    def build_network_graph(self, node_df: pd.DataFrame, edge_df: pd.DataFrame) -> nx.DiGraph:
        """
        Build directed graph for propagation simulation.
        
        Args:
            node_df: Node features
            edge_df: Edge list with source, target
        
        Returns:
            NetworkX directed graph
        """
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for idx, node in node_df.iterrows():
            G.add_node(idx, **node.to_dict())
        
        # Add edges with weights
        for _, edge in edge_df.iterrows():
            source, target = edge['source'], edge['target']
            # Edge weight = dependency strength (higher = stronger dependency)
            weight = edge.get('weight', 1.0)
            G.add_edge(source, target, weight=weight)
        
        print(f"Network graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def calculate_network_centrality(self, G: nx.DiGraph) -> Dict[int, Dict[str, float]]:
        """
        Calculate network centrality measures for all nodes.
        
        Returns:
            Dict mapping node_id to centrality metrics
        """
        print("Calculating network centrality measures...")
        
        centrality = {}
        
        # Degree centrality (in + out)
        in_degree = dict(G.in_degree())
        out_degree = dict(G.out_degree())
        
        # Betweenness centrality (nodes on critical paths)
        betweenness = nx.betweenness_centrality(G)
        
        # PageRank (importance in network)
        pagerank = nx.pagerank(G)
        
        for node in G.nodes():
            centrality[node] = {
                'in_degree': in_degree[node],
                'out_degree': out_degree[node],
                'betweenness': betweenness[node],
                'pagerank': pagerank[node]
            }
        
        return centrality
    
    def propagate_disruption(
        self,
        G: nx.DiGraph,
        initial_nodes: List[int],
        initial_severity: Dict[int, float],
        node_buffers: Dict[int, float]
    ) -> Dict[int, float]:
        """
        Propagate disruption through network using cascading failure model.
        
        Args:
            G: Supply chain network graph
            initial_nodes: Initially disrupted nodes
            initial_severity: Severity for each initial node [0.3, 1.0]
            node_buffers: Buffer capacity for each node [0, 1]
        
        Returns:
            Dict mapping node_id to final impact severity
        """
        # Track affected nodes and their impact
        affected = {node: initial_severity[node] for node in initial_nodes}
        
        # Propagate for max_hops iterations
        for hop in range(self.max_hops):
            new_affected = {}
            
            for node in list(affected.keys()):
                # Get all downstream nodes (successors)
                for successor in G.successors(node):
                    if successor in affected:
                        continue  # Already affected
                    
                    # Calculate propagated impact
                    edge_weight = G[node][successor].get('weight', 1.0)
                    buffer = node_buffers.get(successor, 0.5)
                    
                    # Impact formula: severity × edge_weight × decay × (1 - buffer)
                    propagated_impact = (
                        affected[node] * 
                        edge_weight * 
                        self.propagation_decay * 
                        (1 - buffer)
                    )
                    
                    # Only propagate if impact is significant (> 0.1)
                    if propagated_impact > 0.1:
                        if successor not in new_affected:
                            new_affected[successor] = propagated_impact
                        else:
                            # Multiple sources affecting same node - take max
                            new_affected[successor] = max(
                                new_affected[successor],
                                propagated_impact
                            )
            
            # Add newly affected nodes
            affected.update(new_affected)
            
            if len(new_affected) == 0:
                break  # No more propagation
        
        return affected
    
    def generate_targeted_scenarios(
        self,
        node_df: pd.DataFrame,
        edge_df: pd.DataFrame,
        G: nx.DiGraph,
        centrality: Dict[int, Dict[str, float]],
        node_buffers: Dict[int, float]
    ) -> List[Dict]:
        """
        Generate 4 targeted disruption scenarios as per research paper.
        
        Scenarios:
        1. Random Supplier Failure (10 random suppliers)
        2. High-Capacity Supplier Failure (5 highest-capacity suppliers)
        3. High-Risk Manufacturer Failure (8 high-risk manufacturers)
        4. Central Distributor Failure (4 distributors with highest betweenness centrality)
        
        Args:
            node_df: Node features
            edge_df: Edge list
            G: Network graph
            centrality: Network centrality measures
            node_buffers: Node buffer capacities
        
        Returns:
            List of 4 targeted scenarios
        """
        scenarios = []
        fixed_severity = 0.7  # 70% reliability reduction as per paper
        
        print("\n" + "="*70)
        print("GENERATING 4 TARGETED DISRUPTION SCENARIOS (Research Paper)")
        print("="*70)
        
        # Scenario 1: Random Supplier Failure (10 random suppliers)
        print("\n[Scenario 1] Random Supplier Failure")
        suppliers = node_df[node_df['tier'] == 0].index.tolist()
        if len(suppliers) >= 10:
            random_suppliers = np.random.choice(suppliers, size=10, replace=False).tolist()
        else:
            random_suppliers = suppliers
        
        initial_severity_1 = {node_id: fixed_severity for node_id in random_suppliers}
        all_affected_1 = self.propagate_disruption(G, random_suppliers, initial_severity_1, node_buffers)
        
        scenarios.append({
            'scenario_id': 'targeted_1_random_supplier',
            'scenario_type': 'Random Supplier Failure',
            'initial_nodes': random_suppliers,
            'all_affected_nodes': list(all_affected_1.keys()),
            'severities': all_affected_1,
            'propagation_count': len(all_affected_1) - len(random_suppliers)
        })
        print(f"  Initial disruptions: {len(random_suppliers)} suppliers")
        print(f"  Total affected: {len(all_affected_1)} nodes")
        print(f"  Propagation: {len(all_affected_1) - len(random_suppliers)} additional nodes")
        
        # Scenario 2: High-Capacity Supplier Failure (5 highest-capacity suppliers)
        print("\n[Scenario 2] High-Capacity Supplier Failure")
        supplier_df = node_df[node_df['tier'] == 0]
        if len(supplier_df) >= 5:
            top_capacity_suppliers = supplier_df.nlargest(5, 'capacity').index.tolist()
        else:
            top_capacity_suppliers = supplier_df.index.tolist()
        
        initial_severity_2 = {node_id: fixed_severity for node_id in top_capacity_suppliers}
        all_affected_2 = self.propagate_disruption(G, top_capacity_suppliers, initial_severity_2, node_buffers)
        
        scenarios.append({
            'scenario_id': 'targeted_2_high_capacity_supplier',
            'scenario_type': 'High-Capacity Supplier Failure',
            'initial_nodes': top_capacity_suppliers,
            'all_affected_nodes': list(all_affected_2.keys()),
            'severities': all_affected_2,
            'propagation_count': len(all_affected_2) - len(top_capacity_suppliers)
        })
        print(f"  Initial disruptions: {len(top_capacity_suppliers)} high-capacity suppliers")
        print(f"  Total affected: {len(all_affected_2)} nodes")
        print(f"  Propagation: {len(all_affected_2) - len(top_capacity_suppliers)} additional nodes")
        
        # Scenario 3: High-Risk Manufacturer Failure (8 high-risk manufacturers)
        print("\n[Scenario 3] High-Risk Manufacturer Failure")
        manufacturers = node_df[node_df['tier'] == 1]
        if len(manufacturers) >= 8:
            high_risk_manufacturers = manufacturers.nlargest(8, 'risk_level').index.tolist()
        else:
            high_risk_manufacturers = manufacturers.index.tolist()
        
        initial_severity_3 = {node_id: fixed_severity for node_id in high_risk_manufacturers}
        all_affected_3 = self.propagate_disruption(G, high_risk_manufacturers, initial_severity_3, node_buffers)
        
        scenarios.append({
            'scenario_id': 'targeted_3_high_risk_manufacturer',
            'scenario_type': 'High-Risk Manufacturer Failure',
            'initial_nodes': high_risk_manufacturers,
            'all_affected_nodes': list(all_affected_3.keys()),
            'severities': all_affected_3,
            'propagation_count': len(all_affected_3) - len(high_risk_manufacturers)
        })
        print(f"  Initial disruptions: {len(high_risk_manufacturers)} high-risk manufacturers")
        print(f"  Total affected: {len(all_affected_3)} nodes")
        print(f"  Propagation: {len(all_affected_3) - len(high_risk_manufacturers)} additional nodes")
        
        # Scenario 4: Central Distributor Failure (4 distributors with highest betweenness)
        print("\n[Scenario 4] Central Distributor Failure")
        distributors = node_df[node_df['tier'] == 2].index.tolist()
        
        # Sort distributors by betweenness centrality
        distributor_centrality = [(node_id, centrality[node_id]['betweenness']) 
                                   for node_id in distributors]
        distributor_centrality.sort(key=lambda x: x[1], reverse=True)
        
        if len(distributor_centrality) >= 4:
            central_distributors = [node_id for node_id, _ in distributor_centrality[:4]]
        else:
            central_distributors = [node_id for node_id, _ in distributor_centrality]
        
        initial_severity_4 = {node_id: fixed_severity for node_id in central_distributors}
        all_affected_4 = self.propagate_disruption(G, central_distributors, initial_severity_4, node_buffers)
        
        scenarios.append({
            'scenario_id': 'targeted_4_central_distributor',
            'scenario_type': 'Central Distributor Failure',
            'initial_nodes': central_distributors,
            'all_affected_nodes': list(all_affected_4.keys()),
            'severities': all_affected_4,
            'propagation_count': len(all_affected_4) - len(central_distributors)
        })
        print(f"  Initial disruptions: {len(central_distributors)} central distributors")
        print(f"  Total affected: {len(all_affected_4)} nodes")
        print(f"  Propagation: {len(all_affected_4) - len(central_distributors)} additional nodes")
        
        print("\n" + "="*70)
        print("TARGETED SCENARIOS SUMMARY")
        print("="*70)
        for i, scenario in enumerate(scenarios, 1):
            print(f"Scenario {i}: {scenario['scenario_type']}")
            print(f"  Initial: {len(scenario['initial_nodes'])} nodes")
            print(f"  Total affected: {len(scenario['all_affected_nodes'])} nodes")
            print(f"  Propagation multiplier: {len(scenario['all_affected_nodes']) / len(scenario['initial_nodes']):.2f}x")
        
        return scenarios
    
    def generate_cascading_scenarios(
        self,
        node_df: pd.DataFrame,
        edge_df: pd.DataFrame,
        num_scenarios: int = 100,
        initial_disruption_prob: float = 0.15,
        use_mixed_targeted: bool = True
    ) -> List[Dict]:
        """
        Generate disruption scenarios with cascading propagation.
        
        Args:
            node_df: Node features
            edge_df: Edge list
            num_scenarios: Number of scenarios to generate
            initial_disruption_prob: Probability of initial disruption (for random type)
            use_mixed_targeted: If True, generate mixture of 4 targeted types
                               If False, generate random probabilistic scenarios
        
        Returns:
            List of scenarios with cascading effects
        """
        # Build network graph
        G = self.build_network_graph(node_df, edge_df)
        
        # Calculate centrality
        centrality = self.calculate_network_centrality(G)
        
        # Extract node buffers (inventory/backup capacity)
        node_buffers = {}
        for idx, node in node_df.iterrows():
            # Buffer = combination of inventory level and backup capacity
            inventory = node.get('inventory_level', 0.5)
            backup = node.get('backup_capacity', 0.5)
            node_buffers[idx] = (inventory + backup) / 2
        
        scenarios = []
        fixed_severity = 0.7  # 70% reliability reduction for targeted scenarios
        
        # Prepare node lists for targeted scenarios
        suppliers = node_df[node_df['tier'] == 0].index.tolist()
        manufacturers = node_df[node_df['tier'] == 1].index.tolist()
        distributors = node_df[node_df['tier'] == 2].index.tolist()
        
        # Sort for capacity and risk-based selection
        supplier_df = node_df[node_df['tier'] == 0]
        manufacturer_df = node_df[node_df['tier'] == 1]
        
        # Distributor centrality for selection
        distributor_centrality = [(node_id, centrality[node_id]['betweenness']) 
                                   for node_id in distributors]
        distributor_centrality.sort(key=lambda x: x[1], reverse=True)
        
        if use_mixed_targeted:
            print(f"\n" + "="*70)
            print(f"GENERATING {num_scenarios} MIXED TARGETED SCENARIOS")
            print("="*70)
            print("Each scenario randomly selects one of 4 types:")
            print("  1. Random Supplier Failure (10 random suppliers)")
            print("  2. High-Capacity Supplier Failure (5 highest-capacity)")
            print("  3. High-Risk Manufacturer Failure (8 high-risk)")
            print("  4. Central Distributor Failure (4 highest betweenness)")
            print("="*70)
            
            scenario_type_counts = {
                'Random Supplier Failure': 0,
                'High-Capacity Supplier Failure': 0,
                'High-Risk Manufacturer Failure': 0,
                'Central Distributor Failure': 0
            }
            
            for scenario_id in range(num_scenarios):
                # Randomly select one of 4 scenario types
                scenario_type = np.random.choice([
                    'Random Supplier Failure',
                    'High-Capacity Supplier Failure',
                    'High-Risk Manufacturer Failure',
                    'Central Distributor Failure'
                ])
                
                scenario_type_counts[scenario_type] += 1
                
                # Generate scenario based on type
                if scenario_type == 'Random Supplier Failure':
                    # Type 1: Random Supplier Failure (10 random suppliers)
                    if len(suppliers) >= 10:
                        initial_nodes = np.random.choice(suppliers, size=10, replace=False).tolist()
                    else:
                        initial_nodes = suppliers
                
                elif scenario_type == 'High-Capacity Supplier Failure':
                    # Type 2: High-Capacity Supplier Failure (5 highest-capacity)
                    if len(supplier_df) >= 5:
                        initial_nodes = supplier_df.nlargest(5, 'capacity').index.tolist()
                    else:
                        initial_nodes = supplier_df.index.tolist()
                
                elif scenario_type == 'High-Risk Manufacturer Failure':
                    # Type 3: High-Risk Manufacturer Failure (8 high-risk)
                    if len(manufacturer_df) >= 8:
                        initial_nodes = manufacturer_df.nlargest(8, 'risk_level').index.tolist()
                    else:
                        initial_nodes = manufacturer_df.index.tolist()
                
                else:  # Central Distributor Failure
                    # Type 4: Central Distributor Failure (4 highest betweenness)
                    if len(distributor_centrality) >= 4:
                        initial_nodes = [node_id for node_id, _ in distributor_centrality[:4]]
                    else:
                        initial_nodes = [node_id for node_id, _ in distributor_centrality]
                
                # Apply fixed severity and propagate
                initial_severity = {node_id: fixed_severity for node_id in initial_nodes}
                all_affected = self.propagate_disruption(G, initial_nodes, initial_severity, node_buffers)
                
                # Store scenario
                scenario = {
                    'scenario_id': f'mixed_{scenario_id}',
                    'scenario_type': scenario_type,
                    'initial_nodes': initial_nodes,
                    'all_affected_nodes': list(all_affected.keys()),
                    'severities': all_affected,
                    'propagation_count': len(all_affected) - len(initial_nodes)
                }
                
                scenarios.append(scenario)
            
            # Print statistics by type
            print(f"\n" + "="*70)
            print("SCENARIO TYPE DISTRIBUTION")
            print("="*70)
            for stype, count in scenario_type_counts.items():
                percentage = (count / num_scenarios) * 100
                print(f"{stype}: {count} ({percentage:.1f}%)")
            
            # Print overall statistics
            initial_counts = [len(s['initial_nodes']) for s in scenarios]
            total_counts = [len(s['all_affected_nodes']) for s in scenarios]
            prop_counts = [s['propagation_count'] for s in scenarios]
            
            print(f"\n" + "="*70)
            print("OVERALL STATISTICS")
            print("="*70)
            print(f"Total scenarios: {len(scenarios)}")
            print(f"Avg initial disruptions: {np.mean(initial_counts):.1f}")
            print(f"Avg total affected: {np.mean(total_counts):.1f}")
            print(f"Avg propagated nodes: {np.mean(prop_counts):.1f}")
            print(f"Propagation multiplier: {np.mean(total_counts) / np.mean(initial_counts):.2f}x")
            
        else:
            # Original random probabilistic scenarios
            print(f"\nGenerating {num_scenarios} random probabilistic scenarios...")
            
            for scenario_id in range(num_scenarios):
                # Step 1: Select initial disrupted nodes
                initial_nodes = []
                initial_severity = {}
                
                for idx, node in node_df.iterrows():
                    region = node.get('region', 'Unknown')
                    base_prob = self.region_disruption_prob.get(region, initial_disruption_prob)
                    
                    if np.random.random() < base_prob * initial_disruption_prob:
                        initial_nodes.append(idx)
                        
                        # Sample disruption type and severity
                        if region in self.region_disruption_types.index:
                            disruption_type = np.random.choice(
                                self.region_disruption_types.columns,
                                p=self.region_disruption_types.loc[region].values
                            )
                        else:
                            disruption_type = np.random.choice([
                                'Port Congestion', 'Cyber Attack', 'Natural Disaster',
                                'Labor Strike', 'Factory Incident', 'Geopolitical'
                            ])
                        
                        # Sample severity
                        mean_sev = self.disruption_severity_mean.get(disruption_type, 2.5)
                        std_sev = self.disruption_severity_std.get(disruption_type, 1.0)
                        severity = np.clip(np.random.normal(mean_sev, std_sev), 1, 5)
                        severity_normalized = 0.3 + (severity - 1) / 4 * 0.7
                        
                        initial_severity[idx] = severity_normalized
                
                # Step 2: Propagate disruption through network
                all_affected = self.propagate_disruption(
                    G, initial_nodes, initial_severity, node_buffers
                )
                
                # Step 3: Store scenario
                scenario = {
                    'scenario_id': f'random_{scenario_id}',
                    'scenario_type': 'Random Probabilistic',
                    'initial_nodes': initial_nodes,
                    'all_affected_nodes': list(all_affected.keys()),
                    'severities': all_affected,
                    'propagation_count': len(all_affected) - len(initial_nodes)
                }
                
                scenarios.append(scenario)
            
            # Print statistics
            initial_counts = [len(s['initial_nodes']) for s in scenarios]
            total_counts = [len(s['all_affected_nodes']) for s in scenarios]
            prop_counts = [s['propagation_count'] for s in scenarios]
            
            print(f"\nRandom Probabilistic Statistics:")
            print(f"  Avg initial disruptions: {np.mean(initial_counts):.1f}")
            print(f"  Avg total affected: {np.mean(total_counts):.1f}")
            print(f"  Avg propagated nodes: {np.mean(prop_counts):.1f}")
            print(f"  Propagation multiplier: {np.mean(total_counts) / np.mean(initial_counts):.2f}x")
        
        return scenarios, centrality
    
    def calculate_network_aware_resilience(
        self,
        node_df: pd.DataFrame,
        scenarios: List[Dict],
        centrality: Dict[int, Dict[str, float]]
    ) -> pd.DataFrame:
        """
        Calculate resilience scores considering network position and cascading effects.
        
        Args:
            node_df: Node features
            scenarios: Cascading disruption scenarios
            centrality: Network centrality measures
        
        Returns:
            DataFrame with resilience scores and labels
        """
        resilience_scores = {}
        disruption_counts = {}
        propagated_counts = {}
        
        for node_id in node_df.index:
            node_impacts = []
            direct_count = 0
            propagated_count = 0
            
            for scenario in scenarios:
                if node_id in scenario['severities']:
                    severity = scenario['severities'][node_id]
                    
                    # Track if direct or propagated
                    if node_id in scenario['initial_nodes']:
                        direct_count += 1
                    else:
                        propagated_count += 1
                    
                    # Resilience contribution (higher severity = lower resilience)
                    resilience_contribution = 1 - severity
                    node_impacts.append(resilience_contribution)
            
            if len(node_impacts) > 0:
                base_resilience = np.mean(node_impacts)
                
                # Adjust based on network centrality
                # High betweenness = more vulnerable (on critical paths)
                betweenness_penalty = centrality[node_id]['betweenness'] * 0.2
                
                # High PageRank = more important but also more vulnerable
                pagerank_penalty = centrality[node_id]['pagerank'] * 0.1
                
                # Final resilience score
                resilience_scores[node_id] = max(0, base_resilience - betweenness_penalty - pagerank_penalty)
                disruption_counts[node_id] = len(node_impacts)
                propagated_counts[node_id] = propagated_count
            else:
                # No disruptions = high resilience
                resilience_scores[node_id] = 1.0
                disruption_counts[node_id] = 0
                propagated_counts[node_id] = 0
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'node_id': list(resilience_scores.keys()),
            'resilience_score': list(resilience_scores.values()),
            'disruption_count': [disruption_counts[nid] for nid in resilience_scores.keys()],
            'propagated_count': [propagated_counts[nid] for nid in resilience_scores.keys()],
            'betweenness': [centrality[nid]['betweenness'] for nid in resilience_scores.keys()],
            'pagerank': [centrality[nid]['pagerank'] for nid in resilience_scores.keys()]
        })
        
        # Binary classification
        result_df['resilient'] = (result_df['resilience_score'] >= self.resilience_threshold).astype(int)
        
        print(f"\nNetwork-Aware Resilience Statistics:")
        print(f"  Mean: {result_df['resilience_score'].mean():.3f}")
        print(f"  Std: {result_df['resilience_score'].std():.3f}")
        print(f"  Min: {result_df['resilience_score'].min():.3f}")
        print(f"  Max: {result_df['resilience_score'].max():.3f}")
        print(f"\nClass Distribution:")
        print(f"  Resilient (1): {(result_df['resilient'] == 1).sum()} ({(result_df['resilient'] == 1).sum() / len(result_df) * 100:.1f}%)")
        print(f"  Vulnerable (0): {(result_df['resilient'] == 0).sum()} ({(result_df['resilient'] == 0).sum() / len(result_df) * 100:.1f}%)")
        
        return result_df


def main():
    """Test cascading disruption simulation."""
    print("="*70)
    print("CASCADING DISRUPTION SIMULATION TEST")
    print("="*70)
    
    # Load data
    node_df = pd.read_csv('synthetic_nodes.csv')
    edge_df = pd.read_csv('synthetic_edges.csv')
    
    print(f"\nLoaded data:")
    print(f"  Nodes: {len(node_df)}")
    print(f"  Edges: {len(edge_df)}")
    
    # Initialize simulator
    simulator = CascadingDisruptionSimulator(
        resilience_threshold=0.6,
        propagation_decay=0.7,
        max_hops=3,
        seed=42
    )
    
    # Generate cascading scenarios (ADVANCED: 1000 scenarios)
    scenarios, centrality = simulator.generate_cascading_scenarios(
        node_df, edge_df,
        num_scenarios=1000,
        initial_disruption_prob=0.15
    )
    
    # Calculate network-aware resilience
    result_df = simulator.calculate_network_aware_resilience(
        node_df, scenarios, centrality
    )
    
    # Save results
    result_df.to_csv('node_resilience_labels_cascading.csv', index=False)
    print(f"\n✓ Saved: node_resilience_labels_cascading.csv")
    
    # Compare with baseline (if exists)
    try:
        baseline_df = pd.read_csv('node_resilience_labels.csv')
        print(f"\n" + "="*70)
        print("COMPARISON WITH BASELINE")
        print("="*70)
        print(f"\nBaseline (no cascading):")
        print(f"  Resilient: {(baseline_df['resilient'] == 1).sum()} ({(baseline_df['resilient'] == 1).sum() / len(baseline_df) * 100:.1f}%)")
        print(f"  Mean score: {baseline_df['resilience_score'].mean():.3f}")
        
        print(f"\nCascading (with propagation):")
        print(f"  Resilient: {(result_df['resilient'] == 1).sum()} ({(result_df['resilient'] == 1).sum() / len(result_df) * 100:.1f}%)")
        print(f"  Mean score: {result_df['resilience_score'].mean():.3f}")
        
        # Nodes that changed classification
        merged = baseline_df.merge(result_df, on='node_id', suffixes=('_baseline', '_cascading'))
        changed = merged[merged['resilient_baseline'] != merged['resilient_cascading']]
        print(f"\nNodes with changed classification: {len(changed)} ({len(changed)/len(merged)*100:.1f}%)")
        
    except FileNotFoundError:
        print("\nNote: Baseline labels not found for comparison")
    
    print(f"\n" + "="*70)
    print("✓ CASCADING SIMULATION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
