"""
Complete Training Pipeline for Supply Chain Resilience
Uses functions from actual_pipeline_backup for balanced class distribution.

This pipeline:
1. Builds/loads graph using SupplyChainGraphBuilder
2. Generates scenarios using RealisticDisruptionSimulator
3. Trains GNN models using imported architectures
4. Benchmarks against ML models
5. Saves results and trained models

Usage:
    python complete_training_pipeline.py
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import os
import json
from datetime import datetime
from tqdm import tqdm
import sys
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ✅ Import from actual_pipeline_backup
from actual_pipeline_backup.generate_realistic_scenarios import RealisticDisruptionSimulator
from actual_pipeline_backup.generate_edge_disruption_scenarios import EdgeDisruptionSimulator
from actual_pipeline_backup.train_multi_gnn_realistic import (
    GATModel, GCNModel, GraphSAGEModel, GINModel, 
    TransformerConvModel, GINEModel,
    train_epoch, evaluate, load_scenario_data, split_scenarios
)
from actual_pipeline_backup.graph_construction import SupplyChainGraphBuilder


class CompletePipeline:
    """
    Complete training pipeline using actual_pipeline_backup functions.
    """
    
    def __init__(self, seed=42, output_dir='pipeline_output', build_from_scratch=False):
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.output_dir = output_dir
        self.build_from_scratch = build_from_scratch
        os.makedirs(output_dir, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Device: {self.device}")
        
    def log(self, message, level='INFO'):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] [{level}] {message}")
    
    # ========================================================================
    # STEP 1: BUILD/LOAD GRAPH
    # ========================================================================
    
    def build_or_load_graph(self, graph_path='actual_pipeline_backup/supply_chain_graph.pt', 
                           num_nodes=500, num_tiers=4):
        """Build graph from scratch or load existing one."""
        
        if self.build_from_scratch:
            self.log("Building graph from scratch...")
            
            # Use SupplyChainGraphBuilder from backup
            builder = SupplyChainGraphBuilder(
                num_nodes=num_nodes,
                num_tiers=num_tiers,
                seed=self.seed
            )
            
            # Generate graph
            builder.generate_graph()
            
            # Save to output directory
            graph_path = os.path.join(self.output_dir, 'supply_chain_graph.pt')
            builder.save_graph(graph_path)
            
            self.log(f"Graph built and saved to {graph_path}")
        else:
            self.log(f"Using existing graph: {graph_path}")
        
        return graph_path
    
    # ========================================================================
    # STEP 2: GENERATE SCENARIOS
    # ========================================================================
    
    def generate_scenarios(self, graph_path, num_scenarios=1000, scenario_type='node'):
        """Generate disruption scenarios using RealisticDisruptionSimulator."""
        self.log(f"Generating {num_scenarios} {scenario_type} disruption scenarios...")
        
        if scenario_type == 'edge':
            # Use EdgeDisruptionSimulator
            simulator = EdgeDisruptionSimulator(seed=self.seed)
        else:
            # Use RealisticDisruptionSimulator
            simulator = RealisticDisruptionSimulator(seed=self.seed)
        
        # Load preprocessed graph
        from actual_pipeline_backup.generate_realistic_scenarios import load_preprocessed_graph
        node_df, edge_df = load_preprocessed_graph(graph_path)
        
        # Load graph
        G = simulator.load_graph(node_df, edge_df)
        
        # Store graph in simulator
        simulator.G = G
        
        # Calculate base buffers
        base_buffers = simulator.calculate_base_buffers()
        
        # Generate scenarios
        scenarios = simulator.generate_scenarios(G, base_buffers, num_scenarios=num_scenarios)
        
        # Create PyG data objects
        data_objects = simulator.create_pyg_data_objects(G, node_df, edge_df, scenarios)
        
        # Save scenarios
        scenario_dir = os.path.join(self.output_dir, f'scenarios_{scenario_type}')
        simulator.save_scenarios(data_objects, output_dir=scenario_dir)
        
        self.log(f"Scenarios saved to {scenario_dir}/")
        
        return data_objects, scenario_dir
    
    # ========================================================================
    # STEP 3: SPLIT DATA
    # ========================================================================
    
    def split_scenarios(self, data_objects, train_ratio=0.7, val_ratio=0.15):
        """Split scenarios into train/val/test sets."""
        self.log("Splitting scenarios...")
        
        num_scenarios = len(data_objects)
        indices = np.random.permutation(num_scenarios)
        
        train_size = int(num_scenarios * train_ratio)
        val_size = int(num_scenarios * val_ratio)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_scenarios = [data_objects[i] for i in train_indices]
        val_scenarios = [data_objects[i] for i in val_indices]
        test_scenarios = [data_objects[i] for i in test_indices]
        
        self.log(f"Split: {len(train_scenarios)} train, {len(val_scenarios)} val, {len(test_scenarios)} test")
        
        return train_scenarios, val_scenarios, test_scenarios
    
    # ========================================================================
    # STEP 4: CALCULATE CLASS WEIGHTS
    # ========================================================================
    
    def calculate_class_weights(self, train_scenarios):
        """Calculate balanced class weights."""
        self.log("Calculating class weights...")
        
        train_labels = []
        for data in train_scenarios:
            valid_mask = (data.y != -1) & data.train_mask
            train_labels.extend(data.y[valid_mask].numpy())
        
        unique_classes = np.unique(train_labels)
        class_weight_values = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=train_labels
        )
        class_weights = torch.tensor(class_weight_values, dtype=torch.float).to(self.device)
        
        self.log(f"Class weights: {class_weights.cpu().numpy()}")
        
        return class_weights, len(unique_classes)
    
    # ========================================================================
    # STEP 5: TRAIN GNN MODELS
    # ========================================================================
    
    def train_gnn_models(self, train_scenarios, val_scenarios, test_scenarios):
        """Train multiple GNN architectures using imported models."""
        self.log("Training GNN models...")
        
        train_loader = DataLoader(train_scenarios, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_scenarios, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_scenarios, batch_size=32, shuffle=False)
        
        in_channels = train_scenarios[0].x.shape[1]
        class_weights, num_classes = self.calculate_class_weights(train_scenarios)
        
        # Check if edge features are available
        has_edge_features = hasattr(train_scenarios[0], 'edge_attr') and train_scenarios[0].edge_attr is not None
        edge_dim = train_scenarios[0].edge_attr.shape[1] if has_edge_features else 0
        
        # Define models
        models = {
            'GAT': (GATModel(in_channels, hidden_channels=128, num_heads=4, dropout=0.3, num_classes=num_classes), False),
            'GCN': (GCNModel(in_channels, hidden_channels=128, dropout=0.3, num_classes=num_classes), False),
            'GraphSAGE': (GraphSAGEModel(in_channels, hidden_channels=128, dropout=0.3, num_classes=num_classes), False),
            'GIN': (GINModel(in_channels, hidden_channels=128, dropout=0.3, num_classes=num_classes), False),
        }
        
        # Add edge-aware models if edge features available
        if has_edge_features:
            models['TransformerConv'] = (TransformerConvModel(in_channels, edge_dim=edge_dim, hidden_channels=128, num_heads=4, dropout=0.3, num_classes=num_classes), True)
            models['GINE'] = (GINEModel(in_channels, edge_dim=edge_dim, hidden_channels=128, dropout=0.3, num_classes=num_classes), True)
            self.log("Added edge-aware models: TransformerConv, GINE")
        
        results = []
        loss_fn = nn.NLLLoss(weight=class_weights)
        
        for model_name, (model, use_edge_attr) in models.items():
            self.log(f"Training {model_name}...")
            
            model = model.to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
            
            best_val_f1 = 0
            patience_counter = 0
            patience = 30
            
            for epoch in range(1, 201):
                # Use imported train_epoch function
                train_loss, train_acc = train_epoch(model, train_loader, optimizer, self.device, loss_fn, use_edge_attr)
                
                # Validation every 10 epochs
                if epoch % 10 == 0:
                    # Use imported evaluate function
                    val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(model, val_loader, self.device, use_edge_attr)
                    
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        patience_counter = 0
                        torch.save(model.state_dict(), os.path.join(self.output_dir, f'best_{model_name.lower()}_model.pt'))
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            break
            
            # Test evaluation
            model.load_state_dict(torch.load(os.path.join(self.output_dir, f'best_{model_name.lower()}_model.pt')))
            test_loss, test_acc, test_prec, test_rec, test_f1, test_preds, test_labels = evaluate(model, test_loader, self.device, use_edge_attr)
            
            self.log(f"{model_name} - Test Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
            
            results.append({
                'model': model_name,
                'accuracy': test_acc,
                'precision': test_prec,
                'recall': test_rec,
                'f1': test_f1
            })
        
        return results
    
    # ========================================================================
    # STEP 6: BENCHMARK ML MODELS
    # ========================================================================
    
    def benchmark_ml_models(self, train_scenarios, test_scenarios):
        """Benchmark traditional ML models."""
        self.log("Benchmarking ML models...")
        
        # Prepare flat features
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        
        for data in train_scenarios:
            valid_mask = (data.y != -1) & data.train_mask
            if valid_mask.sum() > 0:
                X_train.append(data.x[valid_mask, :6].numpy())  # Exclude buffer
                y_train.append(data.y[valid_mask].numpy())
        
        for data in test_scenarios:
            valid_mask = (data.y != -1) & data.train_mask
            if valid_mask.sum() > 0:
                X_test.append(data.x[valid_mask, :6].numpy())
                y_test.append(data.y[valid_mask].numpy())
        
        X_train = np.vstack(X_train)
        y_train = np.concatenate(y_train)
        X_test = np.vstack(X_test)
        y_test = np.concatenate(y_test)
        
        # Train ML models
        ml_models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        }
        
        results = []
        
        for model_name, model in ml_models.items():
            self.log(f"Training {model_name}...")
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            test_acc = accuracy_score(y_test, y_pred)
            test_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            self.log(f"{model_name} - Test Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
            
            results.append({
                'model': model_name,
                'accuracy': test_acc,
                'f1': test_f1
            })
        
        return results
    
    # ========================================================================
    # STEP 7: SAVE RESULTS
    # ========================================================================
    
    def save_results(self, gnn_results, ml_results, num_scenarios, num_nodes, num_edges):
        """Save all results to output directory."""
        self.log("Saving results...")
        
        # Combine results
        all_results = gnn_results + ml_results
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(os.path.join(self.output_dir, 'model_comparison.csv'), index=False)
        
        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'num_scenarios': num_scenarios,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'best_gnn': max(gnn_results, key=lambda x: x['f1']),
            'best_ml': max(ml_results, key=lambda x: x['f1']),
        }
        
        with open(os.path.join(self.output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.log(f"Results saved to {self.output_dir}/")
        
        return df_results
    
    # ========================================================================
    # MAIN PIPELINE
    # ========================================================================
    
    def run(self, graph_path='actual_pipeline_backup/supply_chain_graph.pt', 
            num_scenarios=1000, scenario_type='node', num_nodes=500, num_tiers=4):
        """Run complete pipeline."""
        self.log("="*70)
        self.log("COMPLETE TRAINING PIPELINE")
        self.log("="*70)
        
        # Step 1: Build or load graph
        graph_path = self.build_or_load_graph(graph_path, num_nodes, num_tiers)
        
        # Step 2: Generate scenarios
        data_objects, scenario_dir = self.generate_scenarios(graph_path, num_scenarios, scenario_type)
        
        # Step 3: Split data
        train_scenarios, val_scenarios, test_scenarios = self.split_scenarios(data_objects)
        
        # Step 4: Train GNN models
        gnn_results = self.train_gnn_models(train_scenarios, val_scenarios, test_scenarios)
        
        # Step 5: Benchmark ML models
        ml_results = self.benchmark_ml_models(train_scenarios, test_scenarios)
        
        # Step 6: Save results
        results = self.save_results(gnn_results, ml_results, num_scenarios, 
                                    len(data_objects[0].x), data_objects[0].edge_index.shape[1])
        
        self.log("="*70)
        self.log("PIPELINE COMPLETE!")
        self.log("="*70)
        
        print("\n📊 Final Results:")
        print(results.to_string(index=False))
        
        return results


def main():
    """Main execution."""
    pipeline = CompletePipeline(seed=42, output_dir='pipeline_output', build_from_scratch=False)
    results = pipeline.run(
        graph_path='actual_pipeline_backup/supply_chain_graph.pt',
        num_scenarios=1000,
        scenario_type='node'  # or 'edge' for edge disruptions
    )
    
    print("\n✅ Pipeline completed successfully!")
    print(f"📁 Results saved to: pipeline_output/")


if __name__ == "__main__":
    main()
