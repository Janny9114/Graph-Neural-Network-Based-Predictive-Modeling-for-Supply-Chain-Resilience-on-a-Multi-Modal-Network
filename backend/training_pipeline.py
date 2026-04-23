"""
Training Pipeline for Custom Company Graphs
Refactored version of generate_realistic_scenarios.py and train_multi_gnn_realistic.py
to work with uploaded company data.
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F
import networkx as nx
from collections import deque
import os
import json
from datetime import datetime
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append('C:/Users/janny/Desktop/final_year')
# Import from complete_training_pipeline instead
from complete_training_pipeline import CompletePipeline


class CustomGraphTrainer:
    """
    Trains a custom GNN model on company-specific supply chain data.
    Uses CompletePipeline from complete_training_pipeline.py
    """
    
    def __init__(self, company_dir, progress_callback=None):
        """
        Args:
            company_dir: Directory containing nodes.csv and edges.csv
            progress_callback: Function to call with progress updates (progress, message)
        """
        self.company_dir = company_dir
        self.progress_callback = progress_callback
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize CompletePipeline
        self.pipeline = CompletePipeline(seed=42, output_dir=company_dir)
        
    def update_progress(self, progress, message, stage=''):
        """Update progress if callback is provided."""
        if self.progress_callback:
            self.progress_callback(progress, message, stage)
        print(f"[{progress}%] {message}")
    
    def load_data(self):
        """Load nodes and edges CSV files."""
        self.update_progress(5, "Loading graph data...", "loading")
        
        nodes_path = os.path.join(self.company_dir, 'nodes.csv')
        edges_path = os.path.join(self.company_dir, 'edges.csv')
        
        self.node_df = pd.read_csv(nodes_path)
        self.edge_df = pd.read_csv(edges_path)
        
        self.update_progress(10, f"Loaded {len(self.node_df)} nodes, {len(self.edge_df)} edges", "loaded")
        
        return self.node_df, self.edge_df
    
    def build_graph(self):
        """Build NetworkX graph from data."""
        self.update_progress(12, "Building graph structure...", "graph")
        
        G = nx.DiGraph()
        
        # Add nodes
        for idx, row in self.node_df.iterrows():
            G.add_node(
                idx,
                capacity=row['capacity'],
                risk_level=row['risk_level'],
                reliability=row['reliability'],
                cost_factor=row['cost_factor'],
                tier=row['tier'],
                x=row['x'],
                y=row['y']
            )
        
        # Add edges with lead_time
        for _, row in self.edge_df.iterrows():
            G.add_edge(
                int(row['source']),
                int(row['target']),
                capacity_share=row['capacity_share'],
                lead_time=row.get('lead_time', 7.0)  # Default 7 days if not specified
            )
        
        self.G = G
        self.update_progress(15, f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", "graph_done")
        
        return G
    
    def calculate_buffers(self):
        """
        Calculate safety stock buffer for each node using demand and lead time uncertainty.
        Based on the formula: Safety Stock = Z * sqrt(LT * σ_D² + D² * σ_LT²)
        """
        self.update_progress(17, "Calculating safety stock buffers...", "buffers")
        
        import math
        
        buffers = {}
        for node in self.G.nodes():
            capacity = self.G.nodes[node]['capacity']
            reliability = self.G.nodes[node]['reliability']
            risk_level = self.G.nodes[node]['risk_level']
            
            # Derive parameters from node attributes with randomness
            # Average demand = sum of outgoing capacity (what the node typically ships out)
            successors = list(self.G.successors(node))
            if successors:
                # Sum of outgoing capacity_share (total demand from downstream)
                outgoing_capacity = sum(
                    self.G[node][succ].get('capacity_share', 0.5) 
                    for succ in successors
                )
                avg_demand = capacity * outgoing_capacity  # Scale by node capacity
            else:
                # No outgoing edges (end node/customer) - use full capacity
                avg_demand = capacity * np.random.uniform(0.5, 1.2)
            
            # Standard deviation of demand (10-30% of capacity, higher for less reliable nodes)
            demand_variability = 0.1 + (0.2 * (1 - reliability))  # 10-30%
            std_dev_demand = avg_demand * demand_variability * np.random.uniform(0.8, 1.2)
            
            # Average lead time (days) - from incoming edges
            # Calculate weighted average of incoming edge lead times
            predecessors = list(self.G.predecessors(node))
            if predecessors:
                # Get lead times from incoming edges
                incoming_lead_times = []
                incoming_weights = []
                for pred in predecessors:
                    edge_data = self.G[pred][node]
                    # Read lead_time directly from edge (should be in days)
                    lead_time = edge_data.get('lead_time', 7.0)  # Default 7 days if not specified
                    capacity_share = edge_data.get('capacity_share', 1.0)
                    
                    incoming_lead_times.append(lead_time)
                    incoming_weights.append(capacity_share)
                
                # Weighted average by capacity_share
                total_weight = sum(incoming_weights)
                if total_weight > 0:
                    avg_lead_time = sum(lt * w for lt, w in zip(incoming_lead_times, incoming_weights)) / total_weight
                else:
                    avg_lead_time = np.mean(incoming_lead_times)
            else:
                # No incoming edges (source node) - use default
                avg_lead_time = 7.0  # Default 7 days for source nodes
            
            # Add randomness (±20%)
            avg_lead_time *= np.random.uniform(0.8, 1.2)
            
            # Standard deviation of lead time (20-50% of avg, higher for risky nodes)
            lead_time_variability = 0.2 + (0.3 * risk_level)  # 20-50%
            std_dev_lead_time = avg_lead_time * lead_time_variability * np.random.uniform(0.8, 1.2)
            
            # Z-score (service level) - higher reliability = higher service level
            # 95% service level (Z=1.65) to 99.9% (Z=3.09)
            z_score = 1.65 + (reliability * 1.44)  # 1.65-3.09 range
            z_score *= np.random.uniform(0.9, 1.1)  # Add slight randomness
            
            # Calculate safety stock using the formula
            # Safety Stock = Z * sqrt(LT * σ_D² + D² * σ_LT²)
            demand_variance_component = avg_lead_time * (std_dev_demand ** 2)
            lead_time_variance_component = (avg_demand ** 2) * (std_dev_lead_time ** 2)
            total_variance = demand_variance_component + lead_time_variance_component
            
            safety_stock = z_score * math.sqrt(total_variance)
            
            # Ensure minimum buffer (at least 10% of capacity)
            safety_stock = max(safety_stock, capacity * 0.1)
            
            # Cap maximum buffer (at most 80% of capacity)
            safety_stock = min(safety_stock, capacity * 0.8)
            
            buffers[node] = math.ceil(safety_stock)
        
        self.buffers = buffers
        
        # Calculate statistics
        buffer_values = list(buffers.values())
        avg_buffer = np.mean(buffer_values)
        min_buffer = np.min(buffer_values)
        max_buffer = np.max(buffer_values)
        
        self.update_progress(
            20, 
            f"Calculated safety stock buffers: avg={avg_buffer:.1f}, range=[{min_buffer:.1f}, {max_buffer:.1f}]", 
            "buffers_done"
        )
        
        return buffers
    
    def generate_scenarios(self, num_scenarios=10000):
        """Generate EDGE disruption scenarios (not node disruptions)."""
        self.update_progress(22, f"Generating {num_scenarios} EDGE disruption scenarios...", "scenarios")
        
        scenarios = []
        num_edges = len(self.edge_df)
        
        for i in range(num_scenarios):
            # Update progress every 1000 scenarios
            if i % 1000 == 0 and i > 0:
                progress = 22 + int((i / num_scenarios) * 28)
                self.update_progress(progress, f"Generated {i}/{num_scenarios} scenarios...", "scenarios")
            
            # Random EDGE disruption (1-10% of edges)
            num_disrupted = np.random.randint(1, max(2, num_edges // 10))
            disrupted_edge_indices = np.random.choice(num_edges, size=num_disrupted, replace=False)
            
            # Create edge disruption info
            disrupted_edges = []
            edge_capacity_reduction = {}
            
            for idx in disrupted_edge_indices:
                source = int(self.edge_df.iloc[idx]['source'])
                target = int(self.edge_df.iloc[idx]['target'])
                edge = (source, target)
                disrupted_edges.append(edge)
                
                # Random capacity reduction (30-100%)
                reduction = np.random.uniform(0.3, 1.0)
                edge_capacity_reduction[edge] = reduction
            
            # Simulate edge cascade
            labels = self._simulate_edge_cascade(disrupted_edges, edge_capacity_reduction)
            
            scenarios.append({
                'disrupted_edges': disrupted_edges,
                'edge_capacity_reduction': {str(k): v for k, v in edge_capacity_reduction.items()},
                'labels': labels.tolist()
            })
        
        self.scenarios = scenarios
        self.update_progress(50, f"Generated {len(scenarios)} edge disruption scenarios", "scenarios_done")
        
        return scenarios
    
    def _simulate_edge_cascade(self, disrupted_edges, edge_capacity_reduction):
        """
        Simulate cascading failures when EDGES are disrupted.
        Based on generate_edge_disruption_scenarios.py logic.
        """
        num_nodes = len(self.node_df)
        
        # Add stochastic variance to buffers
        buffers = {
            node: self.buffers[node] * np.random.uniform(0.8, 1.2)
            for node in range(num_nodes)
        }
        
        # Find all nodes affected by edge disruptions
        affected_nodes = set()
        for edge in disrupted_edges:
            affected_nodes.add(edge[1])  # Target nodes directly affected
        
        # Track time to recovery
        time_to_recovery_map = {}
        for edge in disrupted_edges:
            target = edge[1]
            ttr = np.random.uniform(3, 30)  # 3-30 days
            if target not in time_to_recovery_map:
                time_to_recovery_map[target] = ttr
            else:
                time_to_recovery_map[target] = max(time_to_recovery_map[target], ttr)
        
        # Calculate supply shortage for each affected node
        from collections import defaultdict, deque
        shortage_accumulator = defaultdict(float)
        processed = set()
        queue = deque([(node, 0.0, 0) for node in affected_nodes])
        
        labels = np.full(num_nodes, 2)  # Start with all Normal
        
        while queue:
            current_node, accumulated_shortage, depth = queue.popleft()
            
            # Calculate supply shortage due to edge disruptions
            total_inflow = 0
            reduced_inflow = 0
            
            for _, edge_row in self.edge_df.iterrows():
                if int(edge_row['target']) == current_node:
                    predecessor = int(edge_row['source'])
                    edge = (predecessor, current_node)
                    original_weight = edge_row['capacity_share']
                    
                    if edge in edge_capacity_reduction:
                        reduction = edge_capacity_reduction[edge]
                        reduced_weight = original_weight * (1 - reduction)
                        total_inflow += original_weight
                        reduced_inflow += reduced_weight
                    else:
                        total_inflow += original_weight
                        reduced_inflow += original_weight
            
            # Calculate shortage percentage
            if total_inflow > 0:
                shortage_pct = (total_inflow - reduced_inflow) / total_inflow
            else:
                shortage_pct = 0.0
            
            # Add accumulated shortage from upstream
            total_shortage = min(1.0, shortage_pct + accumulated_shortage)
            shortage_accumulator[current_node] += total_shortage
            
            # Process node only once
            if current_node not in processed:
                processed.add(current_node)
                
                # Get node properties
                capacity = self.node_df.iloc[current_node]['capacity']
                buffer = buffers[current_node]
                
                # Time dynamics
                time_to_recovery = time_to_recovery_map.get(current_node, 7.0)
                time_factor = time_to_recovery / 30.0
                
                # Calculate impact
                impact_units = capacity * shortage_accumulator[current_node] * time_factor
                remaining_impact = max(0, impact_units - buffer)
                
                # Determine label
                if impact_units > 0:
                    impact_ratio = remaining_impact / impact_units
                else:
                    impact_ratio = 0.0
                
                # 3-class labels
                if remaining_impact == 0:
                    labels[current_node] = 2  # Normal
                elif impact_ratio < 0.6:
                    labels[current_node] = 1  # Degraded
                else:
                    labels[current_node] = 0  # Failed
                
                # Propagate to downstream nodes
                for _, edge_row in self.edge_df.iterrows():
                    if int(edge_row['source']) == current_node:
                        successor = int(edge_row['target'])
                        if successor not in processed:
                            queue.append((successor, total_shortage * 0.5, depth + 1))
        
        return labels
    
    def create_pyg_datasets(self):
        """Convert scenarios to PyTorch Geometric datasets."""
        self.update_progress(52, "Converting scenarios to graph format...", "converting")
        
        data_list = []
        
        for i, scenario in enumerate(self.scenarios):
            if i % 1000 == 0 and i > 0:
                progress = 52 + int((i / len(self.scenarios)) * 18)
                self.update_progress(progress, f"Converted {i}/{len(self.scenarios)} scenarios...", "converting")
            
            data = self._create_pyg_data(scenario)
            data_list.append(data)
        
        # Split train/val/test
        train_size = int(0.7 * len(data_list))
        val_size = int(0.15 * len(data_list))
        
        self.train_data = data_list[:train_size]
        self.val_data = data_list[train_size:train_size + val_size]
        self.test_data = data_list[train_size + val_size:]
        
        self.update_progress(70, f"Created datasets: {len(self.train_data)} train, {len(self.val_data)} val, {len(self.test_data)} test", "converting_done")
        
        return self.train_data, self.val_data, self.test_data
    
    def _create_pyg_data(self, scenario):
        """Create PyG Data object from EDGE disruption scenario."""
        num_nodes = len(self.node_df)
        disrupted_edges = scenario['disrupted_edges']
        edge_capacity_reduction = {eval(k): v for k, v in scenario['edge_capacity_reduction'].items()}
        labels = torch.tensor(scenario['labels'], dtype=torch.long)
        
        # Node features (11 dimensions) - base features without disruption
        base_features = torch.tensor(
            self.node_df[['capacity', 'cost_factor', 'risk_level', 'reliability', 'x', 'y']].values,
            dtype=torch.float
        ).clone()
        
        # Tier encoding
        tier_encoding = torch.zeros((num_nodes, 4), dtype=torch.float)
        for idx, tier in enumerate(self.node_df['tier'].values):
            tier_encoding[idx, int(tier)] = 1.0
        
        # Buffer feature (add to match expected 11 dimensions)
        buffer_feature = torch.zeros((num_nodes, 1), dtype=torch.float)
        for idx in range(num_nodes):
            buffer_feature[idx, 0] = self.buffers[idx]
        
        x = torch.cat([base_features, buffer_feature, tier_encoding], dim=1)
        
        # Edge index
        edge_index = torch.tensor(
            np.array([self.edge_df['source'].values, self.edge_df['target'].values]),
            dtype=torch.long
        )
        
        # Edge features (4 dimensions) - mark disrupted edges
        num_edges = len(self.edge_df)
        edge_attr = torch.zeros((num_edges, 4), dtype=torch.float)
        
        weights = self.edge_df['capacity_share'].values
        edge_attr[:, 0] = torch.tensor((weights - weights.mean()) / (weights.std() + 1e-8))
        edge_attr[:, 2] = torch.tensor(weights)
        
        for idx, row in self.edge_df.iterrows():
            source_idx = int(row['source'])
            target_idx = int(row['target'])
            edge = (source_idx, target_idx)
            
            # Mark if edge is disrupted
            if edge in edge_capacity_reduction:
                # Edge is disrupted - reduce capacity
                reduction = edge_capacity_reduction[edge]
                edge_attr[idx, 2] *= (1 - reduction)  # Reduce weight
            
            source_cost = float(self.node_df.iloc[source_idx]['cost_factor'])
            target_cost = float(self.node_df.iloc[target_idx]['cost_factor'])
            edge_attr[idx, 1] = (source_cost + target_cost) / 2.0
            
            source_risk = float(self.node_df.iloc[source_idx]['risk_level'])
            target_risk = float(self.node_df.iloc[target_idx]['risk_level'])
            edge_attr[idx, 3] = (source_risk + target_risk) / 2.0
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=labels)
    
    def train_model(self, epochs=100, batch_size=32):
        """Train GNN model."""
        self.update_progress(72, "Initializing model...", "training")
        
        # Create data loaders
        train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_data, batch_size=batch_size)
        
        # Initialize model
        model = GINEModel(in_channels=11, edge_dim=4, hidden_channels=64, dropout=0.3, num_classes=3)
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        
        best_val_acc = 0
        best_model_path = os.path.join(self.company_dir, 'best_model.pt')
        
        self.update_progress(75, f"Training model for {epochs} epochs...", "training")
        
        for epoch in range(epochs):
            # Train
            model.train()
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = F.nll_loss(out, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Validate every 10 epochs
            if epoch % 10 == 0:
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(self.device)
                        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                        pred = out.argmax(dim=1)
                        correct += (pred == batch.y).sum().item()
                        total += batch.y.size(0)
                
                val_acc = correct / total
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), best_model_path)
                
                progress = 75 + int((epoch / epochs) * 20)
                self.update_progress(
                    progress,
                    f"Epoch {epoch}/{epochs} - Loss: {total_loss/len(train_loader):.4f}, Val Acc: {val_acc:.3f}",
                    "training"
                )
        
        self.update_progress(95, f"Training complete! Best validation accuracy: {best_val_acc:.3f}", "training_done")
        
        return best_model_path, best_val_acc
    
    def save_metadata(self, model_path, val_acc):
        """Save training metadata."""
        self.update_progress(97, "Saving metadata...", "saving")
        
        metadata = {
            'company_dir': self.company_dir,
            'num_scenarios': len(self.scenarios),
            'num_nodes': len(self.node_df),
            'num_edges': len(self.edge_df),
            'model_path': model_path,
            'validation_accuracy': float(val_acc),
            'trained_at': datetime.now().isoformat(),
            'device': str(self.device),
            'status': 'completed'
        }
        
        metadata_path = os.path.join(self.company_dir, 'training_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.update_progress(100, "Training pipeline complete!", "done")
        
        return metadata
    
    def load_existing_scenarios(self, scenario_dir='scenario_graphs_edge_disruptions'):
        """Load pre-generated scenarios from disk."""
        self.update_progress(22, "Loading pre-generated edge disruption scenarios...", "loading_scenarios")
        
        # Get full path to scenario directory
        if not os.path.isabs(scenario_dir):
            scenario_dir = os.path.join('C:/Users/janny/Desktop/final_year', scenario_dir)
        
        if not os.path.exists(scenario_dir):
            raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")
        
        # Load metadata
        metadata_path = os.path.join(scenario_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        num_scenarios = metadata['num_scenarios']
        self.update_progress(25, f"Found {num_scenarios} pre-generated scenarios", "loading_scenarios")
        
        # Load scenario files
        data_list = []
        for i in range(num_scenarios):
            if i % 100 == 0 and i > 0:
                progress = 25 + int((i / num_scenarios) * 45)
                self.update_progress(progress, f"Loaded {i}/{num_scenarios} scenarios...", "loading_scenarios")
            
            scenario_path = os.path.join(scenario_dir, f'scenario_{i:05d}.pt')
            if os.path.exists(scenario_path):
                data = torch.load(scenario_path)
                data_list.append(data)
        
        self.update_progress(70, f"Loaded {len(data_list)} scenarios from disk", "loading_done")
        
        # Split train/val/test
        train_size = int(0.7 * len(data_list))
        val_size = int(0.15 * len(data_list))
        
        self.train_data = data_list[:train_size]
        self.val_data = data_list[train_size:train_size + val_size]
        self.test_data = data_list[train_size + val_size:]
        
        self.update_progress(72, f"Split datasets: {len(self.train_data)} train, {len(self.val_data)} val, {len(self.test_data)} test", "split_done")
        
        return self.train_data, self.val_data, self.test_data
    
    def run_full_pipeline(self, num_scenarios=1000, epochs=100, use_existing_scenarios=False, scenario_dir='scenario_graphs_edge_disruptions'):
        """Run the complete training pipeline using CompletePipeline."""
        try:
            self.update_progress(5, "Starting complete training pipeline...", "starting")
            
            # Get paths to uploaded CSV files
            nodes_path = os.path.join(self.company_dir, 'nodes.csv')
            edges_path = os.path.join(self.company_dir, 'edges.csv')
            
            self.update_progress(10, "Running CompletePipeline...", "pipeline")
            
            # Use CompletePipeline.run() with uploaded CSV files
            results = self.pipeline.run(
                node_path=nodes_path,
                edge_path=edges_path,
                num_scenarios=num_scenarios,
                scenario_type='node'  # Use node disruption scenarios
            )
            
            self.update_progress(95, "Pipeline complete, saving results...", "saving")
            
            # Get best model path from pipeline output
            model_path = os.path.join(self.company_dir, 'best_gine_model.pt')
            
            # Extract validation accuracy from results
            gnn_results = [r for r in results.to_dict('records') if 'GNN' in str(r.get('model', ''))]
            if gnn_results:
                best_result = max(gnn_results, key=lambda x: x.get('f1', 0))
                val_acc = best_result.get('accuracy', 0)
            else:
                val_acc = 0
            
            # Save metadata
            metadata = {
                'company_dir': self.company_dir,
                'num_scenarios': num_scenarios,
                'num_nodes': len(pd.read_csv(nodes_path)),
                'num_edges': len(pd.read_csv(edges_path)),
                'model_path': model_path,
                'validation_accuracy': float(val_acc),
                'trained_at': datetime.now().isoformat(),
                'device': str(self.device),
                'status': 'completed',
                'results': results.to_dict('records')
            }
            
            metadata_path = os.path.join(self.company_dir, 'training_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.update_progress(100, "Training pipeline complete!", "done")
            
            return {
                'status': 'success',
                'model_path': model_path,
                'metadata': metadata
            }
            
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            self.update_progress(0, f"Error: {str(e)}", "error")
            return {
                'status': 'error',
                'message': str(e),
                'traceback': error_msg
            }


def train_custom_graph(company_dir, num_scenarios=10000, epochs=100, progress_callback=None, use_existing_scenarios=False, scenario_dir='scenario_graphs_edge_disruptions'):
    """
    Convenience function to train a custom graph.
    
    Args:
        company_dir: Directory containing nodes.csv and edges.csv
        num_scenarios: Number of disruption scenarios to generate (if use_existing_scenarios=False)
        epochs: Number of training epochs
        progress_callback: Function(progress, message, stage) for progress updates
        use_existing_scenarios: If True, load pre-generated scenarios instead of generating new ones
        scenario_dir: Directory containing pre-generated scenarios (if use_existing_scenarios=True)
    
    Returns:
        dict: Training results
    """
    trainer = CustomGraphTrainer(company_dir, progress_callback)
    return trainer.run_full_pipeline(num_scenarios, epochs, use_existing_scenarios, scenario_dir)
