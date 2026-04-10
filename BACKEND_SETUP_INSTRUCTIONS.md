# Backend Setup Instructions

## Overview
This guide explains how to connect the Flask backend API to the Supply Chain Resilience website, enabling real-time GNN-based disruption simulations.

---

## 📋 Prerequisites

- Python 3.8+ installed
- Node.js and npm installed
- All project files in place

---

## 🚀 Step 1: Install Backend Dependencies

Open a terminal in the project directory and run:

```bash
pip install -r backend_requirements.txt
```

This will install:
- Flask (web framework)
- Flask-CORS (cross-origin requests)
- pandas (data processing)
- numpy (numerical operations)
- torch (GNN model)
- networkx (graph operations)

---

## 🔧 Step 2: Start the Backend API Server

In the project root directory (`c:/Users/janny/Desktop/final_year`), run:

```bash
python backend_api.py
```

You should see output like:

```
================================================================================
Supply Chain Resilience API Server
================================================================================
Nodes loaded: 5000
Edges loaded: 7153
GNN model: Available
================================================================================

Starting server on http://localhost:5000
API Endpoints:
  GET  /api/health - Health check
  GET  /api/nodes - Get all nodes
  GET  /api/disruption-types - Get disruption types
  POST /api/simulate - Run simulation
  GET  /api/historical-insights - Get historical data
================================================================================

 * Running on http://0.0.0.0:5000
```

**Keep this terminal open** - the backend server needs to run continuously.

---

## 🌐 Step 3: Start the Frontend Website

Open a **new terminal** and navigate to the website directory:

```bash
cd "Supply Chain Resilience Website (1)"
npm run dev
```

The website will start on `http://localhost:5174/`

---

## ✅ Step 4: Verify the Connection

1. Open your browser to `http://localhost:5174/`
2. Navigate to the **Scenarios** tab
3. You should see the GNN-Based Disruption Impact Simulation
4. If the backend is connected, you'll see disruption types loaded
5. If not connected, you'll see a warning message

---

## 🎯 Available Disruption Scenarios

The backend provides 6 real disruption scenarios:

1. **Random Supplier Failure** - Random selection of suppliers experiencing operational failures
2. **High-Risk Node Targeted** - Disruption targeting nodes with highest inherent risk levels
3. **Critical Path Disruption** - Disruption of nodes with highest network centrality
4. **Regional Disaster** - Natural disaster or geopolitical event affecting a specific region
5. **Port Congestion** - Major shipping port delays affecting cargo movement
6. **Cyber Attack** - Digital infrastructure disruption affecting operations

---

## 🔬 How It Works

### Backend (Python)
- Loads 5000 nodes and 7153 edges from CSV files
- Uses `DisruptionSimulator` class to run cascading disruption analysis
- Implements Equation 30 (Resilience Score) and Equation 2 (Cascading Effects) from the research paper
- Returns real-time simulation results via REST API

### Frontend (React/TypeScript)
- Connects to backend API at `http://localhost:5000/api`
- Sends simulation parameters (node, disruption type, severity, duration)
- Receives and visualizes results:
  - Interactive world map showing affected nodes
  - Resilience score calculation
  - Cascading network effects
  - Impact metrics and mitigation recommendations

---

## 🐛 Troubleshooting

### Backend not starting?
- Check if port 5000 is already in use
- Verify all CSV files are present (`synthetic_nodes.csv`, `synthetic_edges.csv`)
- Ensure Python dependencies are installed

### Frontend can't connect to backend?
- Verify backend is running on `http://localhost:5000`
- Check browser console for CORS errors
- Ensure Flask-CORS is installed

### Simulation not working?
- Check backend terminal for error messages
- Verify CSV data is loaded correctly
- Try different disruption scenarios

---

## 📊 API Endpoints

### GET /api/health
Check backend status
```json
{
  "status": "healthy",
  "nodes": 5000,
  "edges": 7153,
  "gnn_available": true
}
```

### GET /api/disruption-types
Get available disruption scenarios
```json
[
  {
    "id": "random-supplier",
    "name": "Random Supplier Failure",
    "description": "...",
    "severity_range": [0.3, 1.0],
    "typical_duration": [5, 20]
  },
  ...
]
```

### POST /api/simulate
Run disruption simulation
```json
{
  "node_id": "random",
  "disruption_type": "random-supplier",
  "severity": 50,
  "duration": 7
}
```

Returns:
```json
{
  "resilience_score": 0.883,
  "impact_level": "Low Impact",
  "resilient": true,
  "confidence": 89.2,
  "cascading": {
    "affected_nodes": 45,
    "total_nodes": 5000,
    "propagation_percentage": 0.9,
    ...
  },
  "metrics": {
    "estimated_delay": 10,
    "recovery_time": 14,
    "cost_impact": 75000,
    "production_impact": 30
  },
  "affected_nodes": [...],
  "formula": "ρ_i = 1 - (0.50 × 7/30) = 0.883"
}
```

---

## 🎓 Research Paper Integration

The backend implements key concepts from the research paper:

**Equation 30 - Resilience Score:**
```
ρ_i = 1 - (s_d × t_d / t_max)
```
- s_d: Severity (0-1)
- t_d: Duration (days)
- t_max: 30 days (normalization constant)

**Equation 2 - Cascading Effects:**
```
P(v_j | δ_i) = g(P(v_i | δ_i), A_ij, θ_i, θ_j)
```
- 30% severity reduction per hop
- 40% duration reduction per hop
- Network topology-aware propagation

---

## 📝 Notes

- Backend runs on port 5000
- Frontend runs on port 5174
- Both must be running simultaneously
- Backend processes real supply chain graph data
- Simulations use actual network topology and cascading effects
- Results are computed in real-time based on your parameters

---

## 🎉 Success!

If everything is working:
1. Backend shows "Running on http://0.0.0.0:5000"
2. Frontend shows no warning messages
3. Disruption types are loaded in the dropdown
4. Clicking "Run GNN Simulation" shows real results with network visualization

Enjoy exploring supply chain resilience scenarios! 🚀
