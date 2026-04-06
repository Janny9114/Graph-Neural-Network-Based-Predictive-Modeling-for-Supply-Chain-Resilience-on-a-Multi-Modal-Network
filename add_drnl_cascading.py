"""
Add DRNL labels to graph using cascading disruption sources
Uses nodes identified as disruption sources in cascading simulation
"""

import torch
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def calculate_drnl_label(dist_to_sources):
    """
    Calculate DRNL (Double Radius Node Labeling) for a node.
    
    Args:
        dist_to_sources: List of distances to disruption sources
    
    Returns:
        DRNL label (integer)
    """
    if len(dist_to_sources) == 0:
        return 999  # No path to any source
    
    # Get two closest distances
    sorted_dists = sorted(dist_to_sources)
    d_x = sorted_dists[0]
    d_y = sorted_dists[1] if len(sorted_dists) > 1 else sorted_dists[0]
    
    # DRNL formula
    d = d_x + d_y
    label = 1 + min(d_x, d_y) + (d // 2) * ((d // 2) + (d % 2) - 1)
    
    return int(label)


def main():
    print("="*70)
    print("ADDING DRNL LABELS USING CASCADING DISRUPTION SOURCES")
    print("="*70)
    
    # Load graph
    print("\nLoading graph...")
    data = torch.load('supply_chain_graph.pt', weights_only=False)
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.num_edges}")
    print(f"  Features: {data.x.shape}")
    
    # Load cascading labels to identify disruption sources
    print("\nLoading cascading labels...")
    cascading_df = pd.read_csv('node_resilience_labels_cascading.csv')
    
    # Identify disruption sources (nodes with high propagated_count or low resilience)
    # Use nodes that were frequently affected by propagation as indicators
    disruption_sources = cascading_df.nsmallest(20, 'resilience_score')['node_id'].tolist()
    print(f"  Identified {len(disruption_sources)} disruption sources")
    print(f"  Sources: {disruption_sources[:10]}... (showing first 10)")
    
    # Build NetworkX graph
    print("\nBuilding NetworkX graph...")
    G = nx.Graph()
    edge_index = data.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        source, target = edge_index[0, i], edge_index[1, i]
        G.add_edge(int(source), int(target))
    
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Connected: {nx.is_connected(G)}")
    
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        print(f"  Warning: Graph has {len(components)} connected components")
        print(f"  Largest component: {len(max(components, key=len))} nodes")
    
    # Calculate DRNL labels
    print("\nCalculating DRNL labels...")
    drnl_labels = {}
    
    # Only process nodes that exist in the graph
    graph_nodes = set(G.nodes())
    
    for node in range(data.num_nodes):
        if node not in graph_nodes:
            # Node not in graph (isolated) - assign high distance
            drnl_labels[node] = 999
            continue
            
        distances = []
        for source in disruption_sources:
            if source not in graph_nodes:
                continue  # Source not in graph
            try:
                dist = nx.shortest_path_length(G, source=source, target=node)
                distances.append(dist)
            except nx.NetworkXNoPath:
                continue  # No path between source and node
        
        drnl_labels[node] = calculate_drnl_label(distances)
    
    print(f"  Calculated DRNL labels for {len(drnl_labels)} nodes")
    
    # Statistics
    label_values = list(drnl_labels.values())
    print(f"\nDRNL Label Statistics:")
    print(f"  Min: {min(label_values)}")
    print(f"  Max: {max(label_values)}")
    print(f"  Mean: {np.mean(label_values):.2f}")
    print(f"  Std: {np.std(label_values):.2f}")
    print(f"  Unique labels: {len(set(label_values))}")
    
    # Add DRNL as new feature
    drnl_tensor = torch.tensor([drnl_labels[i] for i in range(data.num_nodes)], dtype=torch.float32)
    
    # Normalize DRNL labels (log scale to handle large values)
    drnl_normalized = torch.log1p(drnl_tensor)  # log(1 + x)
    drnl_normalized = (drnl_normalized - drnl_normalized.mean()) / (drnl_normalized.std() + 1e-8)
    
    # Add as new feature column
    data.x = torch.cat([data.x, drnl_normalized.unsqueeze(1)], dim=1)
    
    print(f"\nFeature dimensions:")
    print(f"  Original: torch.Size([{data.num_nodes}, 10])")
    print(f"  With DRNL: {data.x.shape}")
    
    # Save enhanced graph
    torch.save(data, 'supply_chain_graph_cascading_drnl.pt')
    print(f"\n✓ Saved: supply_chain_graph_cascading_drnl.pt")
    
    # Save DRNL labels for analysis
    drnl_df = pd.DataFrame({
        'node_id': list(range(data.num_nodes)),
        'drnl_label': [drnl_labels[i] for i in range(data.num_nodes)],
        'drnl_normalized': drnl_normalized.numpy()
    })
    drnl_df.to_csv('drnl_labels_cascading.csv', index=False)
    print(f"✓ Saved: drnl_labels_cascading.csv")
    
    # Visualization
    print("\nGenerating visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw DRNL distribution
    axes[0].hist(label_values, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('DRNL Label')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('DRNL Label Distribution (Raw)')
    axes[0].grid(True, alpha=0.3)
    
    # Normalized DRNL distribution
    axes[1].hist(drnl_normalized.numpy(), bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Normalized DRNL')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('DRNL Label Distribution (Normalized)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('drnl_distribution_cascading.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: drnl_distribution_cascading.png")
    plt.close()
    
    print("\n" + "="*70)
    print("✓ DRNL LABELS ADDED SUCCESSFULLY!")
    print("="*70)
    print("\nNext step:")
    print("  Run: python test_traingraph.py")
    print("  (Update to use 'supply_chain_graph_cascading_drnl.pt')")


if __name__ == "__main__":
    main()
