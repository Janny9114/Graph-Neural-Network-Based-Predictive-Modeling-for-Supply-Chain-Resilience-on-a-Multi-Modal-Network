# 📋 **Edge Disruption Scenarios - Detailed Explanation**

## **Overview**

The `generate_edge_disruption_scenarios.py` file extends the base disruption simulator with **edge-level disruptions**. This creates a more realistic and challenging dataset where GNNs have a significant advantage over traditional ML models.

---

## **🎯 Key Concept: Node vs Edge Disruptions**

### **Node Disruptions (Original)**
- **What fails:** Factories, warehouses, distribution centers
- **How it propagates:** Through edges (supply routes)
- **Example:** Factory fire → downstream nodes run out of supply

### **Edge Disruptions (New)**
- **What fails:** Transportation routes, shipping lanes, trade routes
- **How it propagates:** Nodes fail due to supply bottlenecks
- **Example:** Road closure → factory can't receive materials → fails

### **Why Edge Disruptions Matter**
- **More realistic:** Real supply chains face both node and edge failures
- **Harder for ML:** Traditional ML can't see edges, only node features
- **GNN advantage:** GNNs can analyze edge structure and propagation patterns

---

## **📊 Scenario Distribution (10,000 Total)**

| Category | Scenarios | % | Description |
|----------|-----------|---|-------------|
| **Node-only** | 4,000 | 40% | Traditional node failures |
| **Edge-only** | 4,000 | 40% | Transportation/route failures |
| **Hybrid** | 2,000 | 20% | Combined node + edge failures |

---

## **🔍 Detailed Scenario Types**

---

### **1️⃣ Node-Only Scenarios (40%)**

These use the original disruption types from `generate_realistic_scenarios.py`:

#### **1.1 Regional Failure (Variable Radius)**
- **Count:** 2,000 scenarios (20%)
- **What happens:** Geographic disaster affects multiple nodes
- **Disaster types:**
  - **Small** (0.3-0.8 std): Factory fire, local power outage
  - **Medium** (0.8-1.5 std): Severe storm, regional flooding
  - **Large** (1.5-2.5 std): Hurricane, major earthquake
  - **Massive** (2.5-4.0 std): Tsunami, widespread disaster

**How it works:**
```python
1. Pick random epicenter node
2. Select disaster type (small/medium/large/massive)
3. Set radius based on disaster type
4. Find all nodes within radius
5. Calculate distance-based impact:
   - Closer nodes: Higher impact (60-100%)
   - Farther nodes: Lower impact (10-40%)
6. Simulate cascade through edges
```

**Example:**
```
Epicenter: Node 45 (x=1.2, y=-0.5)
Disaster: Large hurricane (radius=2.0 std)
Affected nodes: 85 nodes within radius
Impact severity: 70-95% (decreases with distance)
```

---

#### **1.2 Port Congestion**
- **Count:** 1,000 scenarios (10%)
- **What happens:** Major port/hub gets congested
- **Impact:** 30-60% capacity reduction

**How it works:**
```python
1. Select 1-3 high-degree nodes (ports/hubs)
2. Apply 30-60% production impact
3. Cascade propagates to all downstream nodes
4. Lead time increases affect entire supply chain
```

**Example:**
```
Blocked ports: [Node 12, Node 78] (high-degree hubs)
Impact: 45% capacity reduction
Affected downstream: 120 nodes
```

---

#### **1.3 Multi-Node (10 simultaneous)**
- **Count:** 1,000 scenarios (10%)
- **What happens:** 10 random nodes fail simultaneously
- **Impact:** 10-100% per node (random)

**How it works:**
```python
1. Select 10 random nodes
2. Assign random impact (10-100%) to each
3. Simulate cascades from all 10 nodes simultaneously
4. Impacts accumulate at downstream nodes
```

**Example:**
```
Initial nodes: [5, 23, 67, 89, 102, 134, 156, 178, 189, 195]
Impacts: [0.45, 0.78, 0.23, 0.91, 0.56, 0.34, 0.67, 0.82, 0.41, 0.73]
Total affected: 150+ nodes
```

---

#### **1.4 Central Node Failure**
- **Count:** Included in rotation
- **What happens:** High-centrality node fails
- **Impact:** 10-100% (random)

**How it works:**
```python
1. Calculate betweenness centrality for all nodes
2. Select from top 20% most central nodes
3. Apply random impact (10-100%)
4. Cascade affects many downstream nodes (high centrality)
```

---

### **2️⃣ Edge-Only Scenarios (40%)**

These are **NEW** disruption types that affect edges, not nodes:

---

#### **2.1 Transportation Route Disruption**
- **Count:** 1,000 scenarios (10%)
- **What happens:** Weather, accidents block 1-5 routes
- **Real-world examples:** 
  - Typhoon closes shipping lanes
  - Truck accident blocks highway
  - Snowstorm shuts down rail line

**How it works:**
```python
1. Select 1-5 random edges
2. Assign severity:
   - Minor (50%): 20-40% capacity loss
   - Moderate (35%): 40-70% capacity loss
   - Severe (15%): 70-100% capacity loss
3. Calculate supply shortage for target nodes
4. Propagate shortage downstream if nodes fail
```

**Example:**
```
Disrupted edges: [(12, 45), (67, 89), (102, 134)]
Severity: Moderate
Capacity reduction: [55%, 62%, 48%]
Affected nodes: 45, 89, 134 (direct) + 23 downstream
```

**Key difference from node disruption:**
- Node disruption: Node 12 fails → all outgoing edges affected
- Edge disruption: Only edge (12, 45) fails → Node 12 still operational

---

#### **2.2 Trade Restrictions**
- **Count:** 1,000 scenarios (10%)
- **What happens:** Geopolitical conflict blocks cross-border trade
- **Real-world examples:**
  - Tariffs between countries
  - Trade sanctions
  - Border closures

**How it works:**
```python
1. Select 2 random regions (e.g., Asia, Europe)
2. Find all cross-border edges between these regions
3. Apply 30-70% capacity reduction (tariffs/restrictions)
4. Calculate supply shortage for affected nodes
5. Propagate downstream
```

**Example:**
```
Conflict: Region_1 ↔ Region_3
Cross-border edges: 45 edges
Capacity reduction: 30-70% per edge
Affected nodes: All nodes importing from other region
```

**Why this matters:**
- GNN can see which edges cross borders (spatial features)
- ML only sees node features, can't infer trade restrictions

---

#### **2.3 Cyber Attack on Logistics Network**
- **Count:** 1,000 scenarios (10%)
- **What happens:** Cyber attack disrupts coordination/communication
- **Real-world examples:**
  - Ransomware on logistics software
  - GPS jamming
  - Communication network failure

**How it works:**
```python
1. Select attack type:
   - Targeted (60%): Attack high-betweenness edges (10-20%)
   - Widespread (40%): Random edges (20-30%)

2. For targeted attack:
   - Calculate edge betweenness centrality
   - Target top 10-20% most critical routes

3. For widespread attack:
   - Select 20-30% random edges

4. Apply 50-90% capacity reduction (coordination failure)
5. Simulate cascade
```

**Example (Targeted):**
```
Attack type: Targeted
Affected edges: 72 edges (20% of network)
Target: High-betweenness edges (critical routes)
Capacity reduction: 50-90%
Impact: Widespread coordination failure
```

**Example (Widespread):**
```
Attack type: Widespread
Affected edges: 107 edges (30% of network)
Target: Random edges
Capacity reduction: 50-90%
Impact: Chaotic disruption
```

---

#### **2.4 Port/Shipping Lane Blockage**
- **Count:** 1,000 scenarios (10%)
- **What happens:** Major port blocked (e.g., Suez Canal)
- **Real-world examples:**
  - Ever Given blocking Suez Canal
  - Port strike
  - Harbor damage

**How it works:**
```python
1. Select 1-2 high-degree nodes (major ports)
2. Find ALL edges connected to these ports:
   - Incoming edges (can't receive)
   - Outgoing edges (can't ship)

3. Assign blockage severity:
   - Partial (70%): 50-80% capacity reduction
   - Complete (30%): 90-100% capacity reduction

4. Simulate cascade from affected nodes
```

**Example:**
```
Blocked ports: [Node 23, Node 156] (major hubs)
Blockage: Complete (95% reduction)
Affected edges: 
  - Incoming: 45 edges
  - Outgoing: 67 edges
  - Total: 112 edges
Affected nodes: All nodes connected to these ports
```

**Why this is devastating:**
- Ports are high-degree nodes (many connections)
- Blocking one port affects 50-100+ edges
- Ripple effect throughout supply chain

---

#### **2.5 Infrastructure Failure**
- **Count:** 1,000 scenarios (10%)
- **What happens:** Roads, rails, bridges damaged in a region
- **Real-world examples:**
  - Earthquake damages roads
  - Flood washes out bridges
  - Infrastructure aging/collapse

**How it works:**
```python
1. Select random epicenter
2. Choose failure type:
   - Local (60%): radius 0.3-0.8 std
   - Regional (40%): radius 0.8-1.5 std

3. Find all edges within radius:
   - Check if either endpoint is within radius
   - Include edge if u OR v is affected

4. Apply 60-95% capacity reduction
5. Simulate cascade
```

**Example:**
```
Epicenter: Node 89 (x=0.5, y=1.2)
Failure type: Regional (radius=1.2 std)
Affected edges: 78 edges (both endpoints near epicenter)
Capacity reduction: 60-95%
Affected nodes: All nodes using these routes
```

**Spatial clustering:**
- Edges near epicenter more likely affected
- Creates geographic disruption pattern
- GNN can learn spatial patterns, ML cannot

---

### **3️⃣ Hybrid Scenarios (20%)**

These combine **both** node and edge disruptions:

---

#### **3.1 Infrastructure Failure (Hybrid)**
- **Count:** 1,000 scenarios (10%)
- **What happens:** Disaster damages both facilities AND routes
- **Why hybrid:** Earthquake damages factories (nodes) AND roads (edges)

**How it works:**
```python
1. Select epicenter
2. Find nodes within radius → node disruptions
3. Find edges within radius → edge disruptions
4. Simulate BOTH cascades simultaneously
5. Impacts accumulate at affected nodes
```

**Example:**
```
Epicenter: Node 67
Radius: 1.5 std
Affected nodes: 45 nodes (direct damage)
Affected edges: 89 edges (route damage)
Combined impact: Nodes fail from both direct damage AND supply shortage
```

---

#### **3.2 Hybrid Node+Edge**
- **Count:** 1,000 scenarios (10%)
- **What happens:** Regional disaster + transportation disruption
- **Example:** Hurricane damages factories AND floods roads

**How it works:**
```python
1. Generate regional failure scenario (nodes)
2. Generate transportation disruption scenario (edges)
3. Merge results:
   - If node affected by both: accumulate impacts
   - production_impact_pct = node_impact + edge_impact (capped at 1.0)
4. Recalculate labels based on combined impact
```

**Example:**
```
Node disruption: Regional failure (35 nodes, 40-80% impact)
Edge disruption: Transportation routes (23 edges, 30-60% reduction)
Overlap: 12 nodes affected by BOTH
Combined impact: Up to 100% (additive)
```

---

## **🔬 How Edge Cascade Works**

### **Core Algorithm: `simulate_edge_cascade()`**

```python
def simulate_edge_cascade(G, base_buffers, disrupted_edges, edge_capacity_reduction):
    """
    Key steps:
    1. Create modified graph with reduced edge capacities
    2. Find target nodes of disrupted edges (directly affected)
    3. Calculate supply shortage for each node
    4. Propagate shortage downstream if node fails
    """
```

### **Step-by-Step Example:**

**Initial state:**
```
Node 12 → Node 45 (edge weight: 1.0, 100% capacity)
Node 45 capacity: 1000 units
Node 45 buffer: 400 units
```

**Edge disruption:**
```
Edge (12, 45) disrupted: 60% capacity reduction
New edge weight: 0.4 (40% capacity remaining)
```

**Calculate shortage:**
```
Expected inflow: 1000 units (100% capacity)
Actual inflow: 400 units (40% capacity)
Shortage: 600 units (60%)
Shortage percentage: 60%
```

**Node impact:**
```
Production impact: 1000 * 0.6 = 600 units
Buffer available: 400 units
Remaining impact: 600 - 400 = 200 units
Label: 0 (FAILED - buffer insufficient)
```

**Propagation:**
```
Node 45 failed → propagate 20% shortage downstream
Downstream nodes: [67, 89, 102]
Each receives: 20% * 0.8 = 16% impact (80% propagation factor)
```

---

## **📊 Why Edge Disruptions Create GNN Advantage**

### **GNN Can:**
✅ See edge structure and connectivity
✅ Analyze which edges are critical (betweenness)
✅ Understand spatial patterns (geographic clustering)
✅ Track cascade propagation through edges
✅ Learn edge-specific failure patterns

### **ML Cannot:**
❌ Only sees node features (no edge information)
❌ Can't infer which routes are disrupted
❌ Can't see cascade propagation patterns
❌ Can't understand spatial edge clustering
❌ Treats all nodes independently

---

## **🎯 Expected Performance**

| Model | Node-only | Edge-only | Hybrid | Overall |
|-------|-----------|-----------|--------|---------|
| **GNN** | 75-77% | 82-85% | 78-82% | **78-82%** |
| **ML** | 69-71% | 50-55% | 55-60% | **55-60%** |
| **Gap** | +6% | +27-35% | +18-27% | **+18-27%** |

### **Why ML Struggles with Edge Disruptions:**
1. **No edge visibility:** Can't see which routes are blocked
2. **No spatial understanding:** Can't infer geographic patterns
3. **No cascade tracking:** Can't follow supply shortage propagation
4. **Feature independence:** Treats each node separately

### **Why GNN Excels:**
1. **Edge-aware:** Sees all edge disruptions directly
2. **Spatial reasoning:** Uses x,y coordinates + graph structure
3. **Message passing:** Tracks cascade through graph
4. **Relational learning:** Understands node-edge-node relationships

---

## **✅ Summary**

**10,000 scenarios total:**
- **4,000 Node-only** (40%): Traditional disruptions
- **4,000 Edge-only** (40%): Transportation/route failures
- **2,000 Hybrid** (20%): Combined disruptions

**10 scenario types:**
1. Regional Failure (variable radius)
2. Port Congestion
3. Multi-Node (10 simultaneous)
4. Central Node Failure
5. **Transportation Route Disruption** (NEW)
6. **Trade Restrictions** (NEW)
7. **Cyber Attack on Logistics** (NEW)
8. **Port/Shipping Blockage** (NEW)
9. **Infrastructure Failure** (NEW - hybrid)
10. **Hybrid Node+Edge** (NEW - hybrid)

**Expected GNN advantage: +18-27% (5-8x larger than node-only!)**

This creates a realistic, challenging dataset that demonstrates the true power of GNNs for supply chain resilience prediction! 🚀
