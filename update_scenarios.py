"""
Script to update generate_edge_disruption_scenarios.py with new scenario types.
"""

import re

# Read the file
with open('generate_edge_disruption_scenarios.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Replace function definition and body for major_supplier_failure
old_function = '''    def simulate_major_supplier_failure(self, G, base_buffers):
        """
        Scenario 4: Major supplier failure.
        Disrupts 1-3 suppliers with highest capacity, representing critical supplier disruption.
        """
        # Select 1-2 high-degree nodes (ports)
        out_degrees = dict(G.out_degree())
        sorted_nodes = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)
        top_20_pct = max(1, len(sorted_nodes) // 5)
        port_candidates = [n for n, _ in sorted_nodes[:top_20_pct]]
        
        num_ports = np.random.randint(1, 3)
        blocked_ports = np.random.choice(port_candidates, size=min(num_ports, len(port_candidates)), replace=False)
        
        # Find all edges connected to blocked ports
        disrupted_edges = []
        for port in blocked_ports:
            # Outgoing edges (port can't ship)
            for successor in G.successors(port):
                disrupted_edges.append((port, successor))
            # Incoming edges (port can't receive)
            for predecessor in G.predecessors(port):
                disrupted_edges.append((predecessor, port))
        
        # Blockage severity
        blockage_type = np.random.choice(['partial', 'complete'], p=[0.7, 0.3])
        
        edge_capacity_reduction = {}
        for edge in disrupted_edges:
            if blockage_type == 'partial':
                reduction = np.random.uniform(0.5, 0.8)  # 50-80% reduction
            else:  # complete
                reduction = np.random.uniform(0.9, 1.0)  # 90-100% reduction
            
            edge_capacity_reduction[edge] = reduction
        
        # Simulate cascade
        results = self.simulate_edge_cascade(G, base_buffers, disrupted_edges, edge_capacity_reduction)
        
        return {
            'scenario_type': 'port_shipping_blockage',
            'disrupted_edges': disrupted_edges,
            'edge_capacity_reduction': edge_capacity_reduction,
            'blocked_ports': list(blocked_ports),
            'blockage_type': blockage_type,
            'results': results
        }'''

new_function = '''    def simulate_major_supplier_failure(self, G, base_buffers):
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
        }'''

content = content.replace(old_function, new_function)

# 2. Replace scenario type in list
content = content.replace("'port_shipping_blockage',", "'major_supplier_failure',")
content = content.replace("'multi_node_10',", "'random_supplier_failure',")

# 3. Replace scenario call
content = content.replace(
    "elif scenario_type == 'port_shipping_blockage':\n                scenario = self.simulate_port_shipping_blockage(G, base_buffers)",
    "elif scenario_type == 'major_supplier_failure':\n                scenario = self.simulate_major_supplier_failure(G, base_buffers)"
)

# 4. Replace multi_node_10 scenario
old_multi_node = '''            elif scenario_type == 'multi_node_10':
                nodes = list(G.nodes())
                initial_nodes = list(np.random.choice(nodes, size=10, replace=False))
                initial_impact_pcts = [np.random.uniform(0.1, 1.0) for _ in range(10)]
                results = self.simulate_multi_node_cascade(G, base_buffers, initial_nodes, initial_impact_pcts)
                
                scenarios.append({
                    'scenario_id': i,
                    'scenario_type': scenario_type,
                    'disruption_category': 'node_only',
                    'initial_node': initial_nodes,
                    'initial_impact_pct': initial_impact_pcts,
                    'results': results
                })'''

new_random_supplier = '''            elif scenario_type == 'random_supplier_failure':
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
                })'''

content = content.replace(old_multi_node, new_random_supplier)

# Write back
with open('generate_edge_disruption_scenarios.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Successfully updated generate_edge_disruption_scenarios.py")
print("\nChanges made:")
print("  1. ✅ Replaced port_shipping_blockage with major_supplier_failure")
print("  2. ✅ Replaced multi_node_10 with random_supplier_failure")
print("  3. ✅ Updated function implementations")
print("\n⚠️ NEXT CRITICAL STEP: Update parent class (generate_realistic_scenarios.py) to 3-class labeling!")
