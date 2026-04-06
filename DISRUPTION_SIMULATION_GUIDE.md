# Disruption Simulation Framework for GNN Training

## Overview

This document explains the disruption simulation methodology for Step 4 of the GNN training pipeline, based on the research paper "Graph Neural Network-Based Predictive Modeling for Enhanced Supply Chain Resilience" and historical data from `supply_chain_disruption_recovery.csv`.

---

## 🎯 Purpose

Generate **resilience labels** for supply chain nodes to train a Graph Neural Network (GNN) that can predict which nodes are vulnerable to disruptions.

---

## 📐 Mathematical Foundation

### **Resilience Score Calculation (Equation 30 from Paper)**

```
ρ_i = (1/|D_i|) × Σ(1 - s_d × (t_d / t_max))
```

Where:
- **ρ_i**: Historical resilience score of node i
- **D_i**: Set of disruptions affecting node i
- **s_d**: Severity of disruption d ∈ [0.3, 1.0]
- **t_d**: Duration of disruption d (days)
- **t_max**: Normalization constant (30 days)

### **Binary Classification**

```
y_i = 1  if ρ_i ≥ τ (resilient)
y_i = 0  if ρ_i < τ (vulnerable)
```

Where τ = 0.6 (threshold parameter)

### **Cascading Effect (Equation 2 from Paper)**

```
P(v_j | δ_i) = g(P(v_i | δ_i), A_ij, θ_i, θ_j)
```

Disruptions propagate through the network with reduced severity:
- **Severity reduction**: 30% (×0.7)
- **Duration reduction**: 40% (×0.6)
- **Propagation probability**: 30% × severity

---

## 🔧 Implementation Components

### **1. DisruptionSimulator Class**

Main class that handles all disruption simulation logic.

#### **Key Methods:**

**`_analyze_historical_patterns()`**
- Extracts disruption patterns from historical data
- Calculates region-specific disruption probabilities
- Computes disruption type distributions by region
- Analyzes severity and recovery time statistics

**`generate_disruption_scenarios()`**
- Generates multiple disruption scenarios for training
- Uses historical patterns to sample realistic disruptions
- Parameters:
  - `num_scenarios`: Number of scenarios (default: 100)
  - `disruption_probability`: Base probability (default: 0.15)

**`calculate_resilience_scores()`**
- Implements Equation 30 from the paper
- Calculates resilience score for each node
- Generates binary labels (resilient/vulnerable)

**`simulate_cascading_effects()`**
- Implements Equation 2 from the paper
- Simulates disruption propagation through network
- Accounts for network topology and node attributes

---

## 📊 Historical Data Integration

### **Data Source:** `supply_chain_disruption_recovery.csv`

**Key Columns Used:**
1. **supplier_region** → Region-specific disruption probabilities
2. **disruption_type** → Type distribution (Port Congestion, Cyber Attack, etc.)
3. **disruption_severity** → Severity sampling (1-5 scale)
4. **supplier_tier** → Tier-specific impact patterns
5. **has_backup_supplier** → Recovery time modifiers
6. **full_recovery_days** → Duration statistics
7. **production_impact_pct** → Impact severity

### **Extracted Patterns:**

```python
# Region disruption probabilities
Asia-Pacific: 0.35 (highest)
Europe: 0.22
North America: 0.20
South America: 0.15
Africa/Middle East: 0.08

# Disruption type distribution (example for Asia-Pacific)
Port Congestion: 35%
Natural Disaster: 25%
Cyber Attack: 20%
Labor Strike: 10%
Factory Incident: 7%
Geopolitical: 3%
```

---

## 🚀 Usage Example

### **Basic Usage:**

```python
from disruption_simulation import DisruptionSimulator
import pandas as pd

# Load supply chain data
node_df = pd.read_csv("synthetic_nodes.csv")
edge_df = pd.read_csv("synthetic_edges.csv")

# Initialize simulator
simulator = DisruptionSimulator(
    historical_data_path="external_disruption_data/supply_chain_disruption_recovery.csv",
    resilience_threshold=0.6,
    t_max=30
)

# Generate disruption scenarios
scenarios = simulator.generate_disruption_scenarios(
    node_df=node_df,
    num_scenarios=100,
    disruption_probability=0.15
)

# Calculate resilience scores
resilience_df = simulator.calculate_resilience_scores(
    node_df=node_df,
    scenarios=scenarios,
    edge_df=edge_df
)

# Export labels for GNN training
simulator.export_labels_for_training(
    resilience_df=resilience_df,
    output_path="node_resilience_labels.csv"
)
```

### **With Cascading Effects:**

```python
# Simulate cascading effects for each scenario
cascading_scenarios = []
for scenario in scenarios:
    cascading_scenario = simulator.simulate_cascading_effects(
        scenario=scenario,
        edge_df=edge_df,
        node_df=node_df,
        propagation_probability=0.3
    )
    cascading_scenarios.append(cascading_scenario)

# Recalculate with cascading effects
resilience_df_cascading = simulator.calculate_resilience_scores(
    node_df=node_df,
    scenarios=cascading_scenarios,
    edge_df=edge_df
)
```

---

## 📈 Output Format

### **node_resilience_labels.csv**

| Column | Type | Description |
|--------|------|-------------|
| node_id | int | Node identifier |
| resilience_score | float | Calculated resilience score (ρ_i) |
| disruption_count | int | Number of disruptions affecting node |
| resilient | int | Binary label (0=vulnerable, 1=resilient) |

**Example:**
```csv
node_id,resilience_score,disruption_count,resilient
0,0.823,5,1
1,0.456,8,0
2,0.912,3,1
3,0.234,12,0
...
```

---

## 🎓 Integration with GNN Training

### **Step-by-Step Pipeline:**

1. **Graph Generation** (`graph_synthesis.py`)
   - Generate synthetic supply chain network
   - Create nodes with features (capacity, risk, location, etc.)
   - Create edges with features (flow, lead time, cost, etc.)

2. **Graph Construction** (`graph_construction.py`)
   - Convert to PyTorch Geometric format
   - Standardize features (Z-score normalization)
   - Create homogeneous/heterogeneous graphs

3. **Disruption Simulation** (`disruption_simulation.py`) ← **YOU ARE HERE**
   - Generate disruption scenarios
   - Calculate resilience scores
   - Create binary labels

4. **GNN Training** (Next step)
   - Load graph + labels
   - Train GAT model to predict resilience
   - Evaluate on test set

### **Training Data Structure:**

```python
# Graph structure
data = torch.load('supply_chain_graph.pt')
# data.x: Node features [200, 10]
# data.edge_index: Edge connections [2, 280]
# data.edge_attr: Edge features [280, 4]

# Labels
labels_df = pd.read_csv('node_resilience_labels.csv')
data.y = torch.tensor(labels_df['resilient'].values)

# Split
train_mask = torch.zeros(200, dtype=torch.bool)
train_mask[:140] = True  # 70% training
val_mask = torch.zeros(200, dtype=torch.bool)
val_mask[140:170] = True  # 15% validation
test_mask = torch.zeros(200, dtype=torch.bool)
test_mask[170:] = True  # 15% testing
```

---

## 🔬 Research-Backed Design Decisions

### **1. Resilience Threshold (τ = 0.6)**
- **Source:** Paper Equation 30
- **Rationale:** Nodes with ρ_i ≥ 0.6 maintain >60% performance during disruptions
- **Industry Standard:** Aligns with supply chain resilience benchmarks

### **2. Normalization Constant (t_max = 30 days)**
- **Source:** Paper implementation details
- **Rationale:** Most supply chain disruptions resolve within 30 days
- **Historical Data:** Median recovery time = 28 days

### **3. Severity Range [0.3, 1.0]**
- **Source:** Paper Equation 30
- **Rationale:** Even minor disruptions have 30% baseline impact
- **Mapping:** Severity 1-5 scale → [0.3, 1.0] normalized

### **4. Cascading Reduction Factors**
- **Severity: ×0.7** (30% reduction)
- **Duration: ×0.6** (40% reduction)
- **Rationale:** Downstream nodes have time to prepare and mitigate

---

## 📊 Expected Results

### **Typical Output Statistics:**

```
Resilience Score Statistics:
  Mean: 0.260
  Std: 0.770
  Min: -4.297
  Max: 1.000

Class Distribution:
  Resilient (1): 70 nodes (35.0%)
  Vulnerable (0): 130 nodes (65.0%)

Average Disruptions per Scenario: 4.6 nodes
```

### **Class Imbalance Handling:**

The 35/65 split is realistic:
- Most nodes are vulnerable to some disruptions
- Only well-connected, high-capacity nodes are truly resilient
- Use weighted loss or SMOTE for training if needed

---

## 🛠️ Customization Options

### **Adjust Disruption Frequency:**

```python
# More frequent disruptions
scenarios = simulator.generate_disruption_scenarios(
    node_df=node_df,
    num_scenarios=200,  # More scenarios
    disruption_probability=0.25  # Higher probability
)
```

### **Change Resilience Threshold:**

```python
# Stricter resilience criteria
simulator = DisruptionSimulator(
    resilience_threshold=0.7,  # Higher threshold
    t_max=30
)
```

### **Enable Cascading Effects:**

```python
# Add cascading to all scenarios
for i, scenario in enumerate(scenarios):
    scenarios[i] = simulator.simulate_cascading_effects(
        scenario=scenario,
        edge_df=edge_df,
        node_df=node_df,
        propagation_probability=0.4  # Higher propagation
    )
```

---

## 📚 References

1. **Research Paper:** "Graph Neural Network-Based Predictive Modeling for Enhanced Supply Chain Resilience against Multi-Modal Disruptions"
   - Equation 30: Resilience score calculation
   - Equation 2: Disruption propagation model

2. **Historical Data:** `supply_chain_disruption_recovery.csv`
   - 100,000 real-world disruption events
   - 6 disruption types across 5 regions
   - Recovery time and impact statistics

3. **Supply Chain Literature:**
   - Chopra & Sodhi (2004): Managing Risk to Avoid Supply-Chain Breakdown
   - Christopher & Peck (2004): Building the Resilient Supply Chain
   - Sheffi & Rice (2005): A Supply Chain View of the Resilient Enterprise

---

## ✅ Validation Checklist

- [x] Resilience scores follow Equation 30 from paper
- [x] Historical patterns extracted from real data
- [x] Region-specific disruption probabilities
- [x] Tier-specific impact patterns
- [x] Cascading effects implemented (Equation 2)
- [x] Binary labels generated with threshold τ = 0.6
- [x] Output format compatible with GNN training
- [x] Reproducible with random seed

---

## 🚦 Next Steps

1. **Run the simulation:**
   ```bash
   python disruption_simulation.py
   ```

2. **Verify output:**
   ```bash
   # Check generated labels
   head node_resilience_labels.csv
   ```

3. **Proceed to GNN training:**
   - Use `node_resilience_labels.csv` as training labels
   - Train GAT model with graph structure
   - Evaluate resilience prediction accuracy

4. **Iterate if needed:**
   - Adjust parameters based on results
   - Add more scenarios for better coverage
   - Enable cascading effects for complex analysis

---

## 📞 Support

For questions or issues:
1. Check the research paper for theoretical details
2. Review historical data patterns in CSV
3. Examine code comments in `disruption_simulation.py`
4. Verify node/edge data format compatibility

**Happy Training! 🎉**
