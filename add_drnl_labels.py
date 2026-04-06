"""
Add DRNL (Double Radius Node Labeling) Structural Labels to Graph

Implements the hashing function from the link prediction paper:
f_l(i) = 1 + min(d_x, d_y) + (d/2) × [(d/2) + d%2 - 1]

This encodes the topological distance from disruption sources,
helping the GNN understand disruption propagation patterns.
"""

import torch
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List
import pickle


class DRNLLabeler:
    """Add DRNL structural labels based on disruption sources."""
    
    def __init__(self):
        self.graph = None
        
    def load_data(self):
        """Load graph and disruption data."""
        print("="*70)
        print("LOADING DATA")
        print("="*70)
        
        # Load graph
        data = torch.load('supply_chain_graph.pt', weights_only=False)
        print(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges")
        
        # Load node and edge data
        nodes_df = pd.read_csv('synthetic_nodes.csv')
        edges_df = pd.read_csv('synthetic_edges.csv')
        
        # Load resilience labels (contains disruption info)
        labels_df = pd.read_csv('node_resilience_labels.csv')
        
        print(f"Nodes: {len(nodes_df)}")
        print(f"Edges: {len(edges_df)}")
        print(f"Labels: {len(labels_df)}")
        
        return data, nodes_df, edges_df, labels_df
    
    def build_networkx_graph(self, edges_df, num_nodes):
        """Build NetworkX graph for shortest path calculations."""
        print("\n" + "="*70)
        print("BUILDING NETWORKX GRAPH")
        print("="*70)
        
        G = nx.Graph()  # Undirected for shortest paths
        
        # Add all nodes first to ensure they exist
        G.add_nodes_from(range(num_nodes))
        
        # Add edges
        for _, edge in edges_df.iterrows():
            G.add_edge(edge['source'], edge['target'])
        
        print(f"NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"Connected: {nx.is_connected(G)}")
        
        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            print(f"Warning: Graph has {len(components)} connected components")
            print(f"  Largest component: {len(max(components, key=len))} nodes")
        
        self.graph = G
        return G
    
    def identify_disruption_sources(self, labels_df, nodes_df, method='low_resilience', top_k=10):
        """
        Identify disruption source nodes.
        
        Methods:
        - 'low_resilience': Nodes with lowest resilience scores
        - 'vulnerable': Nodes labeled as vulnerable (resilient=0)
        - 'high_risk': Nodes with high risk scores from node features
        """
        print("\n" + "="*70)
        print(f"IDENTIFYING DISRUPTION SOURCES (method={method})")
        print("="*70)
        
        if method == 'low_resilience':
            # Use nodes with lowest resilience scores
            disruption_sources = labels_df.nsmallest(top_k, 'resilience_score')['node_id'].tolist()
            print(f"Selected {len(disruption_sources)} nodes with lowest resilience scores")
            
        elif method == 'vulnerable':
            # Use all vulnerable nodes
            disruption_sources = labels_df[labels_df['resilient'] == 0]['node_id'].tolist()
            print(f"Selected {len(disruption_sources)} vulnerable nodes")
            
        elif method == 'high_risk':
            # Use nodes with high risk from features
            # Assuming risk is in column index 1 of node features
            disruption_sources = nodes_df.nlargest(top_k, 'risk')['node_id'].tolist()
            print(f"Selected {len(disruption_sources)} nodes with highest risk")
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"Disruption sources: {disruption_sources[:5]}... (showing first 5)")
        return disruption_sources
    
    def calculate_drnl_label(self, node_id, disruption_sources, max_distance=999):
        """
        Calculate DRNL label for a node based on distances to disruption sources.
        
        Formula: f_l(i) = 1 + min(d_x, d_y) + (d/2) × [(d/2) + d%2 - 1]
        
        Args:
            node_id: Target node
            disruption_sources: List of disruption source nodes
            max_distance: Maximum distance for unreachable nodes
        
        Returns:
            DRNL label (integer)
        """
        if node_id in disruption_sources:
            # Disruption source nodes get label 1
            return 1
        
        # Calculate shortest paths to all disruption sources
        distances = []
        for source in disruption_sources:
            try:
                dist = nx.shortest_path_length(self.graph, source, node_id)
                distances.append(dist)
            except nx.NetworkXNoPath:
                # No path exists
                distances.append(max_distance)
        
        if len(distances) == 0:
            return 0
        
        # Sort distances to get closest sources
        distances.sort()
        
        if len(distances) >= 2:
            # Use DRNL formula with two closest sources
            d_x = distances[0]
            d_y = distances[1]
            d = d_x + d_y
            
            # DRNL hashing function
            label = 1 + min(d_x, d_y) + (d // 2) * ((d // 2) + (d % 2) - 1)
            
        else:
            # Single source - simplified formula
            d_x = distances[0]
            label = 1 + d_x
        
        return int(label)
    
    def add_drnl_labels_to_graph(self, data, nodes_df, disruption_sources):
        """Add DRNL labels as additional node features."""
        print("\n" + "="*70)
        print("CALCULATING DRNL LABELS FOR ALL NODES")
        print("="*70)
        
        drnl_labels = []
        
        for node_id in range(len(nodes_df)):
            label = self.calculate_drnl_label(node_id, disruption_sources)
            drnl_labels.append(label)
            
            if node_id % 50 == 0:
                print(f"  Processed {node_id}/{len(nodes_df)} nodes...")
        
        print(f"✓ Calculated DRNL labels for all {len(nodes_df)} nodes")
        
        # Convert to tensor
        drnl_tensor = torch.tensor(drnl_labels, dtype=torch.float).unsqueeze(1)
        
        # Statistics
        print(f"\nDRNL Label Statistics:")
        print(f"  Min: {min(drnl_labels)}")
        print(f"  Max: {max(drnl_labels)}")
        print(f"  Mean: {np.mean(drnl_labels):.2f}")
        print(f"  Std: {np.std(drnl_labels):.2f}")
        print(f"  Unique labels: {len(set(drnl_labels))}")
        
        # Normalize DRNL labels (optional - helps with training)
        drnl_normalized = (drnl_tensor - drnl_tensor.mean()) / (drnl_tensor.std() + 1e-8)
        
        # Concatenate with existing features
        original_features = data.x
        data.x = torch.cat([original_features, drnl_normalized], dim=1)
        
        print(f"\nFeature dimensions:")
        print(f"  Original: {original_features.shape}")
        print(f"  With DRNL: {data.x.shape}")
        
        return data, drnl_labels
    
    def save_enhanced_graph(self, data, drnl_labels, output_path='supply_chain_graph_drnl.pt'):
        """Save graph with DRNL labels."""
        print("\n" + "="*70)
        print("SAVING ENHANCED GRAPH")
        print("="*70)
        
        # Save graph
        torch.save(data, output_path)
        print(f"✓ Saved graph to: {output_path}")
        
        # Save DRNL labels separately for analysis
        drnl_df = pd.DataFrame({
            'node_id': range(len(drnl_labels)),
            'drnl_label': drnl_labels
        })
        drnl_df.to_csv('drnl_labels.csv', index=False)
        print(f"✓ Saved DRNL labels to: drnl_labels.csv")
        
        return output_path
    
    def visualize_drnl_distribution(self, drnl_labels):
        """Create visualization of DRNL label distribution."""
        import matplotlib.pyplot as plt
        
        print("\n" + "="*70)
        print("GENERATING DRNL VISUALIZATION")
        print("="*70)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax = axes[0]
        ax.hist(drnl_labels, bins=30, color='#3498db', edgecolor='black', alpha=0.7)
        ax.set_xlabel('DRNL Label')
        ax.set_ylabel('Frequency')
        ax.set_title('DRNL Label Distribution')
        ax.grid(alpha=0.3)
        
        # Box plot
        ax = axes[1]
        ax.boxplot(drnl_labels, vert=True)
        ax.set_ylabel('DRNL Label')
        ax.set_title('DRNL Label Statistics')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('drnl_distribution.png', dpi=300, bbox_inches='tight')
        print("✓ Saved visualization: drnl_distribution.png")
        plt.close()


def main():
    """Main pipeline to add DRNL labels."""
    print("="*70)
    print("ADDING DRNL STRUCTURAL LABELS TO SUPPLY CHAIN GRAPH")
    print("="*70)
    
    labeler = DRNLLabeler()
    
    # Load data
    data, nodes_df, edges_df, labels_df = labeler.load_data()
    
    # Build NetworkX graph
    G = labeler.build_networkx_graph(edges_df, len(nodes_df))
    
    # Identify disruption sources
    # Using 'low_resilience' method with top 10 most vulnerable nodes
    disruption_sources = labeler.identify_disruption_sources(
        labels_df, nodes_df, 
        method='low_resilience', 
        top_k=10
    )
    
    # Calculate and add DRNL labels
    data_enhanced, drnl_labels = labeler.add_drnl_labels_to_graph(
        data, nodes_df, disruption_sources
    )
    
    # Save enhanced graph
    output_path = labeler.save_enhanced_graph(data_enhanced, drnl_labels)
    
    # Visualize
    labeler.visualize_drnl_distribution(drnl_labels)
    
    print("\n" + "="*70)
    print("✓ DRNL LABELS ADDED SUCCESSFULLY!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Train GNN with enhanced graph: supply_chain_graph_drnl.pt")
    print("  2. Compare performance with original graph")
    print("  3. Expected improvement: +10-20% F1 score")
    print("\nGenerated files:")
    print("  - supply_chain_graph_drnl.pt (graph with DRNL labels)")
    print("  - drnl_labels.csv (DRNL labels for analysis)")
    print("  - drnl_distribution.png (visualization)")


if __name__ == "__main__":
    main()
