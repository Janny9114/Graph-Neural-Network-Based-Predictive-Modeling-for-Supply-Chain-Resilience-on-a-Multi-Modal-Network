# Supply Chain Resilience Website

A comprehensive web application for analyzing and predicting supply chain disruptions using Graph Neural Networks (GNNs). This platform enables users to visualize supply chain networks, simulate disruption scenarios, and predict cascading failures using state-of-the-art machine learning models.

---

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [User Guide](#user-guide)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [API Documentation](#api-documentation)
- [Troubleshooting](#troubleshooting)
- [Architecture](#architecture)
- [Optional: Advanced Training](#optional-advanced-training)

---

## 🚀 Quick Start

### Prerequisites Check

Before starting, ensure you have:
- ✅ Python 3.8+ installed
- ✅ Node.js 16+ installed
- ✅ 8GB+ RAM available

### Step 1: Start Backend (Terminal 1)

```bash
# Navigate to project root
cd C:\Users\janny\Desktop\final_year

# Activate virtual environment (if not already activated)
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Start backend server
cd backend
python app.py
```

**✅ Success:** You should see:
```
📦 Loading default GINE model from pipeline_output...
✅ Default model loaded successfully!
 * Running on http://127.0.0.1:5000
```

### Step 2: Start Frontend (Terminal 2)

```bash
# Navigate to frontend directory
cd "C:\Users\janny\Desktop\final_year\Supply Chain Resilience Website"

# Start development server
npm start
```

**✅ Success:** Browser opens automatically to `http://localhost:3000`

### Step 3: Start Using the Application

1. **Use Default Graph** - Click to load pre-trained model (recommended for first-time users)
2. **Select nodes** on the map to simulate disruptions
3. **Click "Predict Impact"** to see results
4. **Explore analytics** dashboard for detailed insights

---

## ✨ Features

### 🎯 Core Capabilities

1. **Interactive Network Visualization**
   - 3D world map with supply chain nodes
   - Real-time node status updates
   - Tier-based color coding (Suppliers, Manufacturers, Distributors, Retailers)
   - Geographic distribution analysis

2. **GNN-Powered Predictions**
   - 6 state-of-the-art GNN models (GAT, GCN, GraphSAGE, GIN, GINE, TransformerConv)
   - Real-time disruption impact prediction
   - Cascading failure analysis
   - 92%+ accuracy on test scenarios

3. **Disruption Simulation**
   - Node disruptions (supplier failures, factory shutdowns)
   - Edge disruptions (transportation route failures, trade restrictions)
   - Hybrid disruptions (combined node + edge failures)
   - Adjustable severity levels (0-100%)

4. **What-If Analysis**
   - Compare multiple disruption scenarios
   - Analyze impact on different supply chain tiers
   - Identify vulnerable nodes and critical paths
   - Export results for reporting

5. **Custom Graph Upload**
   - Upload your own supply chain network
   - Automatic graph preprocessing
   - Model training on custom data
   - Company-specific predictions

6. **Analytics Dashboard**
   - Risk distribution charts
   - Network topology metrics
   - Cascading failure heatmaps
   - Vulnerable node analysis
   - Model comparison tables

---

## 🔧 Prerequisites

### System Requirements

- **Operating System:** Windows 10/11, macOS 10.15+, or Linux
- **RAM:** 8GB minimum (16GB recommended)
- **Storage:** 5GB free space
- **GPU:** Optional (CUDA-compatible GPU for faster training)

### Software Requirements

#### Backend (Python)
- Python 3.8 or higher
- pip (Python package manager)

#### Frontend (React)
- Node.js 16.x or higher
- npm 8.x or higher

---

## 📦 Installation

### Step 1: Clone the Repository

```bash
cd C:\Users\janny\Desktop\final_year
```

### Step 2: Backend Setup

#### 2.1 Create Python Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

#### 2.2 Install Python Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install flask flask-cors pandas numpy scikit-learn matplotlib seaborn networkx tqdm
```

**Note:** Adjust CUDA version (cu118) based on your GPU. Use `cpu` if no GPU available.

#### 2.3 Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
```

### Step 3: Frontend Setup

#### 3.1 Navigate to Frontend Directory

```bash
cd "Supply Chain Resilience Website"
```

#### 3.2 Install Node Dependencies

```bash
npm install
```

#### 3.3 Verify Installation

```bash
npm list react react-dom
```

---

## 🚀 Running the Application

### Option 1: Quick Start (Recommended)

#### Terminal 1: Start Backend

```bash
# From project root: C:\Users\janny\Desktop\final_year
cd backend
python app.py
```

**Expected output:**
```
📦 Loading default GINE model from pipeline_output...
   Detected hidden_channels: 256
✅ Default model loaded successfully!
   Device: cuda
 * Running on http://127.0.0.1:5000
```

#### Terminal 2: Start Frontend

```bash
# From project root: C:\Users\janny\Desktop\final_year
cd "Supply Chain Resilience Website"
npm start
```

**Expected output:**
```
Compiled successfully!

You can now view supply-chain-resilience-website in the browser.

  Local:            http://localhost:3000
  On Your Network:  http://192.168.1.x:3000
```

#### Access the Application

Open your browser and navigate to:
```
http://localhost:3000
```

---

### Option 2: Production Build

#### Build Frontend

```bash
cd "Supply Chain Resilience Website"
npm run build
```

#### Serve Production Build

```bash
# Install serve globally
npm install -g serve

# Serve the build
serve -s build -l 3000
```

---

## 📖 User Guide

### 1. Initial Setup

When you first open the application, you'll see the **Initial Setup** page.

#### Option A: Use Default Graph

1. Click **"Use Default Graph"**
2. The system loads a pre-trained model with 200 nodes
3. You can immediately start making predictions

#### Option B: Upload Custom Graph

1. Click **"Upload Custom Graph"**
2. Prepare two CSV files:

**nodes.csv:**
```csv
node_id,region,tier,capacity,cost_factor,risk_level,reliability,x,y
0,China,0,1000,0.5,0.3,0.9,0.5,0.5
1,USA,1,800,0.6,0.2,0.95,0.2,0.3
...
```

**edges.csv:**
```csv
source,target,capacity_share,lead_time,cost
0,1,0.8,5,100
1,2,0.6,3,80
...
```

3. Upload both files
4. Click **"Process Graph"**
5. Wait for preprocessing (30-60 seconds)
6. System automatically trains a model on your data

---

### 2. Network Visualization

#### World Map View

- **Blue nodes:** Suppliers (Tier 0)
- **Green nodes:** Manufacturers (Tier 1)
- **Yellow nodes:** Distributors (Tier 2)
- **Red nodes:** Retailers (Tier 3)

#### Interactive Features

- **Hover:** View node details (ID, region, tier, capacity)
- **Click:** Select node for disruption simulation
- **Zoom:** Mouse wheel to zoom in/out
- **Rotate:** Click and drag to rotate globe

---

### 3. GNN Prediction

#### Step 1: Select Disruption Type

Choose from three disruption types:

1. **Node Disruption**
   - Simulates supplier failure, factory shutdown, or hub disruption
   - Select one or more nodes to disrupt
   - Adjust severity (0-100%)

2. **Edge Disruption**
   - Simulates transportation route failure or trade restriction
   - Select edges (connections) to disrupt
   - Adjust capacity reduction (0-100%)

3. **Hybrid Disruption**
   - Combines node and edge disruptions
   - Simulates complex scenarios (e.g., earthquake affecting both factories and roads)

#### Step 2: Configure Disruption

**For Node Disruption:**
```
1. Click nodes on the map to select them
2. Selected nodes turn red
3. Adjust severity slider (default: 80%)
4. Click "Predict Impact"
```

**For Edge Disruption:**
```
1. Click two nodes to select an edge
2. Selected edge highlights in red
3. Adjust capacity reduction slider
4. Click "Predict Impact"
```

**For Hybrid Disruption:**
```
1. Select both nodes and edges
2. Adjust both severity sliders
3. Click "Predict Impact"
```

#### Step 3: View Results

The system displays:

- **Prediction Summary:**
  - Failed nodes: X (Y%)
  - Degraded nodes: X (Y%)
  - Normal nodes: X (Y%)

- **Top 10 Most Affected Nodes:**
  - Node ID, Name, Tier
  - Predicted status (Failed/Degraded/Normal)
  - Confidence score (0-100%)

- **Updated Map:**
  - Red: Failed nodes
  - Orange: Degraded nodes
  - Green: Normal nodes

---

### 4. What-If Simulation

Compare multiple disruption scenarios side-by-side.

#### Step 1: Create Scenarios

```
1. Click "Add Scenario"
2. Configure disruption (nodes, edges, severity)
3. Click "Run Scenario"
4. Repeat for up to 5 scenarios
```

#### Step 2: Compare Results

View comparison table:

| Scenario | Failed | Degraded | Normal | Total Impact |
|----------|--------|----------|--------|--------------|
| Scenario 1 | 15 (7.5%) | 35 (17.5%) | 150 (75%) | 25% |
| Scenario 2 | 8 (4%) | 28 (14%) | 164 (82%) | 18% |

#### Step 3: Analyze Insights

- **Best Case:** Scenario with lowest impact
- **Worst Case:** Scenario with highest impact
- **Critical Nodes:** Nodes that fail in multiple scenarios
- **Resilient Nodes:** Nodes that remain normal in all scenarios

---

### 5. Analytics Dashboard

#### Risk Distribution Chart

- Pie chart showing distribution of node statuses
- Breakdown by tier (Suppliers, Manufacturers, etc.)
- Hover for detailed percentages

#### Network Topology Metrics

- **Degree Centrality:** Most connected nodes
- **Betweenness Centrality:** Most critical nodes
- **Clustering Coefficient:** Network density
- **Average Path Length:** Network efficiency

#### Cascading Failure Heatmap

- Visualizes how failures propagate through the network
- X-axis: Time steps
- Y-axis: Nodes
- Color: Failure probability (red = high, green = low)

#### Vulnerable Nodes Analysis

- Lists top 20 most vulnerable nodes
- Vulnerability score (0-100)
- Reasons for vulnerability:
  - High degree (many connections)
  - Low reliability
  - High risk region
  - Critical position in network

---

### 6. Model Comparison

Compare performance of different GNN models.

#### Available Models

1. **GINE** (Graph Isomorphism Network with Edges)
   - Best overall performance (92.2% accuracy)
   - Edge-aware architecture
   - Recommended for most use cases

2. **TransformerConv**
   - Attention-based mechanism
   - Good for complex cascades
   - 91.9% accuracy

3. **GraphSAGE**
   - Sampling-based aggregation
   - Fast inference
   - 91.8% accuracy

4. **GAT** (Graph Attention Network)
   - Attention weights on neighbors
   - 91.4% accuracy

5. **GIN** (Graph Isomorphism Network)
   - Most expressive GNN
   - 91.3% accuracy

6. **GCN** (Graph Convolutional Network)
   - Simple and fast
   - 90.3% accuracy

#### Comparison Metrics

- **Accuracy:** Overall prediction accuracy
- **Precision:** Correctness of positive predictions
- **Recall:** Coverage of actual positives
- **F1 Score:** Harmonic mean of precision and recall
- **Inference Time:** Time to make predictions

---

### 7. Data Management

#### Export Results

```
1. Click "Export Results"
2. Choose format:
   - CSV (for Excel/analysis)
   - JSON (for API integration)
   - PDF (for reporting)
3. Select data to export:
   - Predictions
   - Network metrics
   - Comparison results
4. Click "Download"
```

#### Save Scenarios

```
1. Click "Save Scenario"
2. Enter scenario name
3. Add description (optional)
4. Click "Save"
5. Access saved scenarios from "Load Scenario" menu
```

#### Clear Data

```
1. Click "Clear All Data"
2. Confirm deletion
3. System resets to initial state
```

---

## 🔌 API Documentation

### Base URL

```
http://localhost:5000/api
```

### Endpoints

#### 1. Health Check

```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "model": "GINE",
  "device": "cuda"
}
```

---

#### 2. Get Graph Data

```http
GET /api/graph?company_id=optional_company_id
```

**Response:**
```json
{
  "nodes": [
    {
      "id": 0,
      "name": "Supplier_0",
      "region": "China",
      "tier": 0,
      "capacity": 1000,
      "x": 0.5,
      "y": 0.5
    }
  ],
  "edges": [
    {
      "source": 0,
      "target": 1,
      "capacity_share": 0.8
    }
  ]
}
```

---

#### 3. Predict Disruption Impact

```http
POST /api/predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "company_id": "optional_company_id",
  "disrupted_nodes": [1, 5, 10],
  "disrupted_edges": [[1, 20], [5, 30]],
  "disruption_severity": 0.8,
  "buffer_capacity": 0.5
}
```

**Response:**
```json
{
  "predictions": [
    {
      "node_id": 0,
      "node_name": "Supplier_0",
      "label": 2,
      "label_name": "Normal",
      "probability": [0.05, 0.15, 0.80],
      "tier": 0,
      "region": "China"
    }
  ],
  "summary": {
    "failed": 15,
    "degraded": 35,
    "normal": 150,
    "total_nodes": 200
  },
  "model_info": {
    "model_name": "GINE",
    "company_id": null
  }
}
```

---

#### 4. Upload Custom Graph

```http
POST /api/upload-graph
Content-Type: multipart/form-data
```

**Form Data:**
- `nodes`: CSV file with node data
- `edges`: CSV file with edge data
- `company_id`: Unique identifier for the company

**Response:**
```json
{
  "status": "success",
  "message": "Graph uploaded and processed successfully",
  "company_id": "company_xyz",
  "nodes_count": 200,
  "edges_count": 378
}
```

---

#### 5. Train Model

```http
POST /api/train
Content-Type: application/json
```

**Request Body:**
```json
{
  "company_id": "company_xyz",
  "model_type": "GINE",
  "num_scenarios": 2000,
  "num_epochs": 400
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Model trained successfully",
  "company_id": "company_xyz",
  "model_path": "pipeline_output/company_xyz/best_gine_model.pt",
  "accuracy": 0.922,
  "f1_score": 0.919
}
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Backend Won't Start

**Error:** `ModuleNotFoundError: No module named 'torch'`

**Solution:**
```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Reinstall dependencies
pip install -r requirements.txt
```

---

#### 2. Frontend Won't Start

**Error:** `npm ERR! code ELIFECYCLE`

**Solution:**
```bash
# Delete node_modules and package-lock.json
rm -rf node_modules package-lock.json

# Reinstall dependencies
npm install

# Try starting again
npm start
```

---

#### 3. CORS Errors

**Error:** `Access to fetch at 'http://localhost:5000/api/predict' from origin 'http://localhost:3000' has been blocked by CORS policy`

**Solution:**
```python
# In backend/app.py, ensure CORS is enabled:
from flask_cors import CORS
app = Flask(__name__)
CORS(app)  # This should be present
```

---

#### 4. Model Loading Errors

**Error:** `FileNotFoundError: [Errno 2] No such file or directory: 'best_gine_model.pt'`

**Solution:**
```bash
# Ensure model files exist
ls pipeline_output/best_gine_model.pt

# If missing, train models:
python train_multi_gnn_realistic.py
```

---

#### 5. GPU/CUDA Errors

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
```python
# In backend/app.py, force CPU usage:
device = torch.device('cpu')  # Change from 'cuda' to 'cpu'
```

---

#### 6. Prediction Errors

**Error:** `Predictions are all the same class`

**Solution:**
```bash
# Regenerate scenarios with unmasking:
python generate_edge_disruption_scenarios.py

# Retrain model:
python train_multi_gnn_realistic.py
```

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Visualization│  │  Prediction  │  │  Analytics   │     │
│  │   (D3.js)    │  │   Interface  │  │  Dashboard   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                           │                                  │
│                           │ HTTP/REST API                    │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Backend (Flask)                        │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │   │
│  │  │  API Routes  │  │ Graph Proc.  │  │  Models  │ │   │
│  │  └──────────────┘  └──────────────┘  └──────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           │ PyTorch Geometric                │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         GNN Models (PyTorch)                        │   │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐ │   │
│  │  │ GINE │  │ Trans│  │ SAGE │  │ GAT  │  │ GIN  │ │   │
│  │  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

#### Frontend
- **React 18:** UI framework
- **TypeScript:** Type-safe JavaScript
- **D3.js:** Data visualization
- **Recharts:** Chart library
- **Axios:** HTTP client
- **TailwindCSS:** Styling

#### Backend
- **Flask:** Web framework
- **PyTorch:** Deep learning framework
- **PyTorch Geometric:** Graph neural networks
- **NetworkX:** Graph analysis
- **Pandas:** Data manipulation
- **NumPy:** Numerical computing

#### Models
- **GINE:** Graph Isomorphism Network with Edges
- **TransformerConv:** Transformer-based GNN
- **GraphSAGE:** Sampling-based GNN
- **GAT:** Graph Attention Network
- **GIN:** Graph Isomorphism Network
- **GCN:** Graph Convolutional Network

---

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Contributors

- **Janny** - Lead Developer

---

## 📧 Support

For issues, questions, or suggestions:
- Create an issue on GitHub
- Email: support@supplychainresilience.com

---

## 🎉 Acknowledgments

- PyTorch Geometric team for the excellent GNN library
- React team for the robust frontend framework
- All contributors and testers

---

**Last Updated:** April 26, 2026
