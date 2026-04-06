"""
Disruption Simulation Framework for GNN Training
Based on research paper: "Graph Neural Network-Based Predictive Modeling for Enhanced Supply Chain Resilience"
and historical data from supply_chain_disruption_recovery.csv

This module implements Step 4: Disruption simulation for generating training labels
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import networkx as nx

class DisruptionSimulator:
    """
    Simulates supply chain disruptions based on historical data patterns
    and generates resilience labels for GNN training.
    
    Key Formula from Paper (Equation 30):
    ρ_i = (1/|D_i|) * Σ(1 - s_d * (t_d / t_max))
    
    where:
    - ρ_i: historical resilience score of node i
    - D_i: set of disruptions affecting node i
    - s_d: severity of disruption d ∈ [0.3, 1.0]
    - t_d: duration of disruption d
    - t_max: normalization constant (30 days)
    """
    
    def __init__(
        self,
        historical_data_path: str = "external_disruption_data/supply_chain_disruption_recovery.csv",
        resilience_threshold: float = 0.6,
        t_max: int = 30,
        seed: int = 42
    ):
        """
        Initialize the disruption simulator.
        
        Args:
            historical_data_path: Path to historical disruption data
            resilience_threshold: Threshold τ for binary classification (default: 0.6)
            t_max: Maximum duration for normalization (default: 30 days)
            seed: Random seed for reproducibility
        """
        self.historical_data = pd.read_csv(historical_data_path)
        self.resilience_threshold = resilience_threshold
        self.t_max = t_max
        np.random.seed(seed)
        
        # Extract disruption patterns from historical data
        self._analyze_historical_patterns()
    
    def _analyze_historical_patterns(self):
        """
        Analyze historical data to extract disruption patterns by:
        - Region
        - Industry
        - Supplier tier
        - Company size
        """
        df = self.historical_data
        
        # Disruption probability by region
        self.region_disruption_prob = df.groupby('supplier_region').size() / len(df)
        
        # Disruption type distribution by region
        self.region_disruption_types = df.groupby(['supplier_region', 'disruption_type']).size().unstack(fill_value=0)
        self.region_disruption_types = self.region_disruption_types.div(self.region_disruption_types.sum(axis=1), axis=0)
        
        # Average severity by disruption type
        self.disruption_severity_mean = df.groupby('disruption_type')['disruption_severity'].mean()
        self.disruption_severity_std = df.groupby('disruption_type')['disruption_severity'].std()
        
        # Recovery time statistics
        self.recovery_time_stats = df.groupby(['disruption_type', 'has_backup_supplier']).agg({
            'full_recovery_days': ['mean', 'std']
        })
        
        # Production impact statistics
        self.production_impact_stats = df.groupby(['disruption_type', 'supplier_tier']).agg({
            'production_impact_pct': ['mean', 'std']
        })
        
        print("Historical disruption patterns analyzed:")
        print(f"  - Regions: {len(self.region_disruption_prob)}")
        print(f"  - Disruption types: {len(self.disruption_severity_mean)}")
        print(f"  - Total historical events: {len(df)}")
    
    def generate_disruption_scenarios(
        self,
        node_df: pd.DataFrame,
        num_scenarios: int = 100,
        disruption_probability: float = 0.15
    ) -> List[Dict]:
        """
        Generate multiple disruption scenarios for training.
        
        Args:
            node_df: DataFrame with node information (tier, region, capacity, etc.)
            num_scenarios: Number of disruption scenarios to generate
            disruption_probability: Probability of each node being disrupted
        
        Returns:
            List of disruption scenarios, each containing:
            - disrupted_nodes: List of node IDs
            - disruption_types: Dict mapping node_id to disruption type
            - severities: Dict mapping node_id to severity [0.3, 1.0]
            - durations: Dict mapping node_id to duration (days)
        """
        scenarios = []
        
        for scenario_id in range(num_scenarios):
            scenario = {
                'scenario_id': scenario_id,
                'disrupted_nodes': [],
                'disruption_types': {},
                'severities': {},
                'durations': {},
                'production_impacts': {}
            }
            
            for idx, node in node_df.iterrows():
                # Determine if node is disrupted based on region-specific probability
                region = node.get('region', 'Unknown')
                base_prob = self.region_disruption_prob.get(region, disruption_probability)
                
                if np.random.random() < base_prob * disruption_probability:
                    scenario['disrupted_nodes'].append(idx)
                    
                    # Sample disruption type based on regional patterns
                    if region in self.region_disruption_types.index:
                        disruption_type = np.random.choice(
                            self.region_disruption_types.columns,
                            p=self.region_disruption_types.loc[region].values
                        )
                    else:
                        # Fallback to uniform distribution
                        disruption_type = np.random.choice([
                            'Port Congestion', 'Cyber Attack', 'Natural Disaster',
                            'Labor Strike', 'Factory Incident', 'Geopolitical'
                        ])
                    
                    scenario['disruption_types'][idx] = disruption_type
                    
                    # Sample severity based on disruption type
                    mean_severity = self.disruption_severity_mean.get(disruption_type, 2.5)
                    std_severity = self.disruption_severity_std.get(disruption_type, 1.0)
                    severity = np.clip(
                        np.random.normal(mean_severity, std_severity),
                        1, 5
                    )
                    # Normalize to [0.3, 1.0] range as per paper
                    severity_normalized = 0.3 + (severity - 1) / 4 * 0.7
                    scenario['severities'][idx] = severity_normalized
                    
                    # Sample duration based on severity and backup supplier status
                    has_backup = node.get('has_backup_supplier', False)
                    base_duration = np.random.exponential(scale=15) * severity
                    if has_backup:
                        base_duration *= 0.5  # 50% reduction with backup
                    scenario['durations'][idx] = min(base_duration, 180)  # Cap at 180 days
                    
                    # Sample production impact
                    tier = node.get('tier', 1)
                    if (disruption_type, tier) in self.production_impact_stats.index:
                        impact_mean = self.production_impact_stats.loc[(disruption_type, tier), ('production_impact_pct', 'mean')]
                        impact_std = self.production_impact_stats.loc[(disruption_type, tier), ('production_impact_pct', 'std')]
                    else:
                        impact_mean = 30.0
                        impact_std = 15.0
                    
                    production_impact = np.clip(
                        np.random.normal(impact_mean, impact_std),
                        1.0, 100.0
                    )
                    scenario['production_impacts'][idx] = production_impact
            
            scenarios.append(scenario)
        
        print(f"Generated {num_scenarios} disruption scenarios")
        print(f"  Average disrupted nodes per scenario: {np.mean([len(s['disrupted_nodes']) for s in scenarios]):.1f}")
        
        return scenarios
    
    def calculate_resilience_scores(
        self,
        node_df: pd.DataFrame,
        scenarios: List[Dict],
        edge_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Calculate resilience scores for each node based on disruption scenarios.
        
        Implements Equation 30 from paper:
        ρ_i = (1/|D_i|) * Σ(1 - s_d * (t_d / t_max))
        
        Args:
            node_df: DataFrame with node information
            scenarios: List of disruption scenarios
            edge_df: Optional edge DataFrame for propagation effects
        
        Returns:
            DataFrame with resilience scores and binary labels
        """
        resilience_scores = {}
        disruption_counts = {}
        
        for node_id in node_df.index:
            node_disruptions = []
            
            for scenario in scenarios:
                if node_id in scenario['disrupted_nodes']:
                    severity = scenario['severities'][node_id]
                    duration = scenario['durations'][node_id]
                    
                    # Calculate resilience contribution (Equation 30)
                    resilience_contribution = 1 - severity * (duration / self.t_max)
                    node_disruptions.append(resilience_contribution)
            
            if len(node_disruptions) > 0:
                # Average resilience across all disruptions
                resilience_scores[node_id] = np.mean(node_disruptions)
                disruption_counts[node_id] = len(node_disruptions)
            else:
                # No disruptions = high resilience
                resilience_scores[node_id] = 1.0
                disruption_counts[node_id] = 0
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'node_id': list(resilience_scores.keys()),
            'resilience_score': list(resilience_scores.values()),
            'disruption_count': [disruption_counts[nid] for nid in resilience_scores.keys()]
        })
        
        # Binary classification based on threshold τ
        result_df['resilient'] = (result_df['resilience_score'] >= self.resilience_threshold).astype(int)
        
        print(f"\nResilience Score Statistics:")
        print(f"  Mean: {result_df['resilience_score'].mean():.3f}")
        print(f"  Std: {result_df['resilience_score'].std():.3f}")
        print(f"  Min: {result_df['resilience_score'].min():.3f}")
        print(f"  Max: {result_df['resilience_score'].max():.3f}")
        print(f"\nClass Distribution:")
        print(f"  Resilient (1): {(result_df['resilient'] == 1).sum()} ({(result_df['resilient'] == 1).sum() / len(result_df) * 100:.1f}%)")
        print(f"  Vulnerable (0): {(result_df['resilient'] == 0).sum()} ({(result_df['resilient'] == 0).sum() / len(result_df) * 100:.1f}%)")
        
        return result_df
    
    def simulate_cascading_effects(
        self,
        scenario: Dict,
        edge_df: pd.DataFrame,
        node_df: pd.DataFrame,
        propagation_probability: float = 0.3
    ) -> Dict:
        """
        Simulate cascading disruption effects through the supply chain network.
        
        Based on Equation 2 from paper:
        P(v_j | δ_i) = g(P(v_i | δ_i), A_ij, θ_i, θ_j)
        
        Args:
            scenario: Initial disruption scenario
            edge_df: Edge DataFrame with connections
            node_df: Node DataFrame with attributes
            propagation_probability: Base probability of disruption propagation
        
        Returns:
            Updated scenario with cascading effects
        """
        # Build network graph
        G = nx.DiGraph()
        for _, edge in edge_df.iterrows():
            G.add_edge(edge['source'], edge['target'])
        
        # Track propagated disruptions
        propagated_nodes = set(scenario['disrupted_nodes'])
        new_disruptions = {}
        
        # Iterate through disrupted nodes and propagate
        for node_id in scenario['disrupted_nodes']:
            severity = scenario['severities'][node_id]
            
            # Get downstream neighbors
            if node_id in G:
                for neighbor in G.successors(node_id):
                    if neighbor not in propagated_nodes:
                        # Calculate propagation probability based on severity and edge attributes
                        prop_prob = propagation_probability * severity
                        
                        if np.random.random() < prop_prob:
                            propagated_nodes.add(neighbor)
                            
                            # Reduced severity for propagated disruptions
                            new_disruptions[neighbor] = {
                                'type': scenario['disruption_types'][node_id],
                                'severity': severity * 0.7,  # 30% reduction
                                'duration': scenario['durations'][node_id] * 0.6,  # 40% reduction
                                'source': node_id
                            }
        
        # Update scenario with cascading effects
        for node_id, disruption in new_disruptions.items():
            scenario['disrupted_nodes'].append(node_id)
            scenario['disruption_types'][node_id] = disruption['type']
            scenario['severities'][node_id] = disruption['severity']
            scenario['durations'][node_id] = disruption['duration']
        
        return scenario
    
    def export_labels_for_training(
        self,
        resilience_df: pd.DataFrame,
        output_path: str = "node_resilience_labels.csv"
    ):
        """
        Export resilience labels in format suitable for GNN training.
        
        Args:
            resilience_df: DataFrame with resilience scores and labels
            output_path: Path to save the labels
        """
        resilience_df.to_csv(output_path, index=False)
        print(f"\nResilience labels exported to: {output_path}")


def main():
    """
    Example usage of the disruption simulator.
    """
    print("="*70)
    print("DISRUPTION SIMULATION FOR GNN TRAINING")
    print("="*70)
    
    # Load synthetic supply chain data
    node_df = pd.read_csv("synthetic_nodes.csv")
    edge_df = pd.read_csv("synthetic_edges.csv")
    
    print(f"\nLoaded supply chain data:")
    print(f"  Nodes: {len(node_df)}")
    print(f"  Edges: {len(edge_df)}")
    
    # Initialize simulator
    simulator = DisruptionSimulator(
        historical_data_path="external_disruption_data/supply_chain_disruption_recovery.csv",
        resilience_threshold=0.6,
        t_max=30
    )
    
    # Generate disruption scenarios
    print("\n" + "="*70)
    print("GENERATING DISRUPTION SCENARIOS")
    print("="*70)
    scenarios = simulator.generate_disruption_scenarios(
        node_df=node_df,
        num_scenarios=100,
        disruption_probability=0.15
    )
    
    # Calculate resilience scores
    print("\n" + "="*70)
    print("CALCULATING RESILIENCE SCORES")
    print("="*70)
    resilience_df = simulator.calculate_resilience_scores(
        node_df=node_df,
        scenarios=scenarios,
        edge_df=edge_df
    )
    
    # Export labels
    simulator.export_labels_for_training(
        resilience_df=resilience_df,
        output_path="node_resilience_labels.csv"
    )
    
    print("\n" + "="*70)
    print("✓ DISRUPTION SIMULATION COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Use node_resilience_labels.csv as training labels")
    print("  2. Train GNN model with graph structure + node features")
    print("  3. Predict resilience for new disruption scenarios")


if __name__ == "__main__":
    main()
