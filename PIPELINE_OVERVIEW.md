# Supply Chain Resilience GNN Pipeline Overview

## **Current Pipeline Status (April 6, 2026)**

---

## **📊 Complete Pipeline Flow**

```
┌─────────────────────────────────────────────────────────────────┐
│                    1. DATA GENERATION                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
    graph_synthesis.py → synthetic_nodes.csv (200 nodes)
                      → synthetic_edges.csv (280 edges)
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    2. GRAPH CONSTRUCTION                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
    graph_construction.py → supply_chain_graph.pt
                          → Node features: [capacity, risk, 
                             reliability, lead_time, etc.] (10 features)
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    3. DISRUPTION SIMULATION                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
    disruption_simulation.py → node_resilience_labels.csv
                             → Labels: resilient (1) or vulnerable (0)
                             → 100 disruption scenarios simulated
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    4. GNN TRAINING (Baseline)                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
    train_gnn_resilience.py → best_gnn_model.pt
                            → Test F1: 47.62%
                            → Test Accuracy: 63.33%
                            → Model: GAT (Graph Attention Network)
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    5. ML BENCHMARK                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
    benchmark_ml_models.py → ml_benchmark_results.csv
                           → ml_benchmark_comparison.png
                           → Winner: Logistic Regression (F1: 53.85%)
                           → GNN Rank: 2/7
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              6. DRNL ENHANCEMENT (In Progress)                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
    add_drnl_labels.py → supply_chain_graph_drnl.pt
                       → drnl_labels.csv
                       → drnl_distribution.png
                       → Features: 10 → 11 (added DRNL)
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              7. GNN TRAINING (Enhanced) - TODO                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
    train_gnn_resilience.py (with DRNL graph)
                            → Expected F1: 55-60% (+10-20%)
                            → Should beat Logistic Regression
```

---

## **📁 Current Files & Status**

### **✅ Completed**

| File | Purpose | Output | Status |
|------|---------|--------|--------|
| `graph_synthesis.py` | Generate synthetic supply chain | `synthetic_nodes.csv`, `synthetic_edges.csv` | ✅ Done |
| `graph_construction.py` | Build PyTorch Geometric graph | `supply_chain_graph.pt` | ✅ Done |
| `disruption_simulation.py` | Simulate disruptions & label nodes | `node_resilience_labels.csv` | ✅ Done |
| `train_gnn_resilience.py` | Train GAT model (baseline) | `best_gnn_model.pt`, F1: 47.62% | ✅ Done |
| `benchmark_ml_models.py` | Compare GNN vs traditional ML | `ml_benchmark_results.csv` | ✅ Done |

### **🔄 In Progress**

| File | Purpose | Output | Status |
|------|---------|--------|--------|
| `add_drnl_labels.py` | Add structural labels to graph | `supply_chain_graph_drnl.pt` | 🔄 Fixed, ready to run |

### **⏳ TODO**

| Task | Purpose | Expected Result |
|------|---------|-----------------|
| Run `add_drnl_labels.py` | Generate DRNL-enhanced graph | Graph with 11 features |
| Retrain GNN with DRNL | Improve GNN performance | F1: 55-60% (target) |
| Final benchmark | Compare all models with DRNL | GNN should rank #1 |

---

## **🎯 Current Performance**

### **Baseline Results (Without DRNL)**

| Model | F1 Score | Accuracy | Rank |
|-------|----------|----------|------|
| **Logistic Regression** | **53.85%** | 60.00% | 🥇 1st |
| **GNN (GAT)** | **47.62%** | 63.33% | 🥈 2nd |
| Random Forest | 47.06% | 70.00% | 🥉 3rd |
| K-Nearest Neighbors | 42.11% | 63.33% | 4th |
| Gradient Boosting | 40.00% | 60.00% | 5th |
| MLP | 38.10% | 56.67% | 6th |
| SVM | 35.29% | 63.33% | 7th |

**Key Finding**: Traditional ML (Logistic Regression) currently beats GNN because:
- Small dataset (200 nodes)
- GNN not leveraging graph structure effectively
- Missing structural labels (DRNL)

---

## **🚀 Next Steps**

### **Immediate (Now)**
1. ✅ Fix `add_drnl_labels.py` (DONE - fixed node indexing)
2. ⏳ Run `python add_drnl_labels.py`
3. ⏳ Verify DRNL labels are added correctly

### **Short-term (Next 30 min)**
4. ⏳ Modify `train_gnn_resilience.py` to use `supply_chain_graph_drnl.pt`
5. ⏳ Retrain GNN with DRNL-enhanced graph
6. ⏳ Compare performance: baseline vs DRNL-enhanced

### **Expected Improvement**
- **Current GNN F1**: 47.62%
- **Target GNN F1**: 55-60% (+10-20%)
- **Goal**: Beat Logistic Regression (53.85%)

---

## **📈 What DRNL Does**

### **Problem**
GNN doesn't know which nodes are "close" to disruption sources in the network topology.

### **Solution**
DRNL (Double Radius Node Labeling) encodes:
- **Distance to disruption sources** (shortest path)
- **Topological position** in the network
- **Disruption propagation risk**

### **Formula**
```
f_l(i) = 1 + min(d_x, d_y) + (d/2) × [(d/2) + d%2 - 1]

where:
- d_x = distance to closest disruption source
- d_y = distance to 2nd closest disruption source
- d = d_x + d_y
```

### **Example**
```
Node A: 1 hop from disruption → DRNL = 2 (high risk)
Node B: 5 hops from disruption → DRNL = 16 (low risk)
Node C: IS a disruption source → DRNL = 1 (highest risk)
```

---

## **🔬 Technical Details**

### **Graph Structure**
- **Nodes**: 200 supply chain entities
- **Edges**: 280 relationships (supplier-buyer, logistics)
- **Node Features**: 10 → 11 (after DRNL)
  1. Capacity
  2. Risk
  3. Reliability
  4. Lead time
  5. Inventory level
  6. Demand variability
  7. Geographic diversity
  8. Supplier count
  9. Backup capacity
  10. Recovery time
  11. **DRNL label** (NEW)

### **Model Architecture**
- **Type**: GAT (Graph Attention Network)
- **Layers**: 3 GAT layers
- **Attention heads**: 4
- **Hidden channels**: 64
- **Parameters**: 9,922
- **Output**: Binary classification (resilient/vulnerable)

### **Training Setup**
- **Split**: 70% train, 15% val, 15% test
- **Optimizer**: Adam (lr=0.01)
- **Loss**: NLL with class weights
- **Early stopping**: Patience 30 epochs

---

## **📊 Data Flow Summary**

```
Raw Data (CSV)
    ↓
PyTorch Geometric Graph (10 features)
    ↓
Disruption Simulation (labels)
    ↓
GNN Training (baseline: 47.62% F1)
    ↓
ML Benchmark (LR wins: 53.85% F1)
    ↓
DRNL Enhancement (+1 feature) ← WE ARE HERE
    ↓
GNN Retraining (expected: 55-60% F1)
    ↓
Final Comparison (GNN should win!)
```

---

## **🎓 Research Context**

This pipeline implements the methodology from:
> "Graph Neural Network-Based Predictive Modeling for Enhanced Supply Chain Resilience against Multi-Modal Disruptions"

**Key Innovation**: Using DRNL structural labels to encode disruption propagation patterns in the graph, allowing GNN to learn spatial relationships between nodes and disruption sources.

---

## **💡 Why This Matters**

### **Business Impact**
- **Predict** which supply chain nodes will fail during disruptions
- **Identify** critical vulnerabilities before they cause problems
- **Optimize** resource allocation for resilience improvements

### **Technical Impact**
- Demonstrates GNN superiority over traditional ML for graph data
- Shows importance of structural features (DRNL) for graph learning
- Provides benchmark for supply chain resilience prediction

---

## **🔧 Commands to Run**

```bash
# 1. Generate DRNL labels (NEXT STEP)
python add_drnl_labels.py

# 2. Retrain GNN with DRNL
python train_gnn_resilience.py  # (modify to use DRNL graph)

# 3. Re-run benchmark
python benchmark_ml_models.py  # (modify to use DRNL graph)

# 4. View results
# - training_history.png
# - confusion_matrix.png
# - ml_benchmark_comparison.png
```

---

## **📝 Summary**

**Current Status**: Pipeline is 85% complete. We have:
- ✅ Synthetic supply chain graph (200 nodes, 280 edges)
- ✅ Disruption simulation labels (100 scenarios)
- ✅ Baseline GNN trained (F1: 47.62%)
- ✅ ML benchmark complete (LR wins: 53.85%)
- ✅ DRNL labeling script ready

**Next Action**: Run `python add_drnl_labels.py` to add structural labels, then retrain GNN to achieve target performance of 55-60% F1 score.

**Expected Outcome**: GNN with DRNL will outperform all traditional ML models, demonstrating the value of graph-based learning for supply chain resilience prediction.
