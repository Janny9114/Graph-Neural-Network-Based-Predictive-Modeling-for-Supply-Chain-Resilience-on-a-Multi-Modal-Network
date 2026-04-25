# Architecture Documentation

## Supply Chain Resilience Platform

**Version:** 1.0.0  
**Last Updated:** April 26, 2026

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Component Details](#component-details)
4. [Data Flow](#data-flow)
5. [Technology Stack](#technology-stack)
6. [Model Architecture](#model-architecture)
7. [Deployment](#deployment)
8. [Security](#security)

---

## System Overview

The Supply Chain Resilience Platform is a full-stack web application that combines:
- **Frontend:** React-based interactive dashboard
- **Backend:** Flask REST API for model serving
- **ML Pipeline:** PyTorch Geometric GNN training pipeline
- **Data Layer:** CSV-based graph storage

### Key Design Principles

✅ **Modularity** - Separate concerns (frontend, backend, ML)  
✅ **Scalability** - Stateless API, async training  
✅ **Extensibility** - Plugin architecture for new models  
✅ **Performance** - GPU acceleration, batch processing  
✅ **Usability** - Interactive visualizations, real-time feedback  

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         USER BROWSER                         │
│                    (React Frontend - Port 3000)              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │Dashboard │  │ Network  │  │Prediction│  │ Training │   │
│  │          │  │   Map    │  │          │  │          │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP/REST API
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    BACKEND API SERVER                        │
│                   (Flask - Port 5000)                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              API Endpoints                            │  │
│  │  /predict  /graph  /train  /analyze  /upload        │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                 │
│  ┌─────────────────┐  ┌────────────────┐  ┌─────────────┐ │
│  │  Model Loader   │  │ Graph Builder  │  │  Training   │ │
│  │  (GNN Models)   │  │  (NetworkX)    │  │  Pipeline   │ │
│  └─────────────────┘  └────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    ML TRAINING PIPELINE                      │
│                  (PyTorch Geometric)                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  1. Graph Construction  →  2. Scenario Generation    │  │
│  │  3. Model Training      →  4. Evaluation             │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │
│  │  GINE   │  │   GAT   │  │   GCN   │  │GraphSAGE│      │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      DATA STORAGE                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Graph Data  │  │ Trained      │  │  Training    │     │
│  │  (CSV/PT)    │  │ Models (PT)  │  │  Results     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Frontend (React + TypeScript)

**Location:** `Supply Chain Resilience Website/`

**Key Components:**

```
src/
├── app/
│   ├── App.tsx                    # Main application
│   ├── components/
│   │   ├── Dashboard.tsx          # Overview metrics
│   │   ├── WorldMapNetwork.tsx    # 3D visualization
│   │   ├── GNNPrediction.tsx      # Prediction interface
│   │   ├── NetworkTopology.tsx    # Topology analysis
│   │   ├── VulnerableNodes.tsx    # Vulnerability analysis
│   │   ├── CascadingFailure.tsx   # Cascading simulation
│   │   ├── PipelineRunner.tsx     # Training interface
│   │   └── DataManagement.tsx     # Upload/manage data
│   └── services/
│       └── gnnApi.ts              # API client
└── styles/
    └── globals.css                # Global styles
```

**Technologies:**
- **React 18** - UI framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Recharts** - Data visualization
- **Three.js** - 3D network rendering

**State Management:**
- React hooks (useState, useEffect)
- Context API for global state
- Local storage for persistence

---

### 2. Backend API (Flask)

**Location:** `backend/`

**Key Files:**

```
backend/
├── app.py                  # Main Flask application
├── training_pipeline.py    # Training orchestration
└── uploads/                # User-uploaded data
    └── {company_id}/
        ├── nodes.csv
        ├── edges.csv
        └── best_*_model.pt
```

**API Endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/health` | GET | Health check |
| `/api/graph` | GET | Get graph data |
| `/api/predict` | POST | Run prediction |
| `/api/upload-graph` | POST | Upload data |
| `/api/run-complete-pipeline` | POST | Start training |
| `/api/pipeline-status/<id>` | GET | Check training status |
| `/api/network-vulnerability` | GET | Vulnerability analysis |
| `/api/network-topology` | GET | Topology metrics |
| `/api/cascading-failure` | GET | Cascading simulation |
| `/api/trade-restriction-scenario` | POST | Trade restriction |

**Key Features:**
- **CORS enabled** - Cross-origin requests
- **File upload** - Multipart form data
- **Background tasks** - Threading for long operations
- **Error handling** - Comprehensive error responses
- **Model caching** - Load models once, reuse

---

### 3. ML Training Pipeline

**Location:** Project root

**Key Scripts:**

```
├── complete_training_pipeline.py    # Orchestrator
├── train_multi_gnn_realistic.py     # GNN training
├── benchmark_ml_realistic.py        # ML baselines
├── graph_construction.py            # Graph building
├── graph_preprocessing.py           # Feature engineering
├── generate_realistic_scenarios.py  # Node disruptions
└── generate_edge_disruption_scenarios.py  # Edge disruptions
```

**Pipeline Stages:**

```
1. Graph Construction
   ├── Load CSV files
   ├── Validate data
   ├── Build NetworkX graph
   └── Convert to PyG Data

2. Scenario Generation
   ├── Simulate disruptions
   ├── Calculate propagation
   ├── Label nodes (Failed/Degraded/Normal)
   └── Save scenarios

3. Model Training
   ├── Split data (70/15/15)
   ├── Calculate class weights
   ├── Train 6 GNN models
   ├── Early stopping
   └── Save best models

4. Evaluation
   ├── Test set evaluation
   ├── Benchmark ML models
   ├── Generate comparison table
   └── Save results
```

---

## Data Flow

### Prediction Flow

```
1. User selects disrupted nodes/edges
   ↓
2. Frontend sends POST /api/predict
   {
     "disrupted_nodes": [1, 5, 10],
     "severity": 0.8
   }
   ↓
3. Backend creates PyG Data object
   - Loads graph structure
   - Applies disruption features
   - Normalizes features
   ↓
4. Model inference
   - Forward pass through GNN
   - Softmax for probabilities
   - Argmax for predictions
   ↓
5. Response formatting
   {
     "predictions": [...],
     "summary": {...}
   }
   ↓
6. Frontend visualization
   - Update map colors
   - Show prediction table
   - Display metrics
```

### Training Flow

```
1. User uploads CSV files
   ↓
2. Backend validates data
   - Check required columns
   - Validate ranges
   - Ensure connectivity
   ↓
3. Generate task ID
   ↓
4. Start background thread
   ↓
5. Training pipeline
   - Build graph
   - Generate 2000 scenarios
   - Train 6 models (30-60 min)
   - Evaluate and compare
   ↓
6. Save results
   - Best models (.pt files)
   - Comparison table (CSV)
   - Training metadata (JSON)
   ↓
7. Frontend polls /api/pipeline-status
   - Shows progress bar
   - Updates status message
   ↓
8. Training complete
   - Display results table
   - Enable model download
```

---

## Technology Stack

### Frontend

| Technology | Version | Purpose |
|------------|---------|---------|
| React | 18.2.0 | UI framework |
| TypeScript | 5.0.0 | Type safety |
| Next.js | 13.4.0 | React framework |
| Tailwind CSS | 3.3.0 | Styling |
| Recharts | 2.6.0 | Charts |
| Three.js | 0.153.0 | 3D graphics |

### Backend

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.10+ | Programming language |
| Flask | 2.3.2 | Web framework |
| Flask-CORS | 4.0.0 | CORS handling |
| Werkzeug | 2.3.6 | WSGI utilities |

### Machine Learning

| Technology | Version | Purpose |
|------------|---------|---------|
| PyTorch | 2.0.1 | Deep learning |
| PyTorch Geometric | 2.3.1 | GNN library |
| NumPy | 1.24.3 | Numerical computing |
| Pandas | 2.0.3 | Data manipulation |
| Scikit-learn | 1.3.0 | ML algorithms |
| NetworkX | 3.1 | Graph analysis |

---

## Model Architecture

### GINE (Graph Isomorphism Network with Edge Features)

**Best performing model (76.6% accuracy)**

```python
Input: Node features (11D) + Edge features (4D)
  ↓
GINEConv Layer 1 (11 → 256)
  ↓ ReLU + BatchNorm + Dropout(0.3)
GINEConv Layer 2 (256 → 256)
  ↓ ReLU + BatchNorm + Dropout(0.3)
GINEConv Layer 3 (256 → 256)
  ↓ ReLU + BatchNorm + Dropout(0.3)
GINEConv Layer 4 (256 → 256)
  ↓ ReLU + BatchNorm + Dropout(0.3)
Fully Connected (256 → 3)
  ↓ Log Softmax
Output: [P(Failed), P(Degraded), P(Normal)]
```

**Key Features:**
- **Edge-aware:** Uses edge features for better predictions
- **Deep:** 4 layers for complex pattern learning
- **Regularized:** Dropout + BatchNorm prevent overfitting
- **Balanced:** Square-root class weighting

### Node Features (11 dimensions)

1. **Capacity** (normalized) - Production/storage capacity
2. **Cost Factor** (normalized) - Operational cost
3. **Risk Level** (normalized) - Historical vulnerability
4. **Reliability** (normalized) - Uptime percentage
5. **X coordinate** (normalized) - Geographic longitude
6. **Y coordinate** (normalized) - Geographic latitude
7-10. **Tier encoding** (one-hot) - Supply chain level
11. **Is Disrupted** (binary) - Disruption flag

### Edge Features (4 dimensions)

1. **Lead Time** (normalized) - Shipping duration
2. **Cost** (normalized) - Transportation cost
3. **Capacity Share** (normalized) - Allocated capacity
4. **Disruption Probability** (0-1) - Edge disruption flag

---

## Deployment

### Development Setup

```bash
# 1. Clone repository
git clone <repo-url>
cd final_year

# 2. Install Python dependencies
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Install Node dependencies
cd "Supply Chain Resilience Website"
npm install

# 4. Start backend
cd ../backend
python app.py  # Runs on port 5000

# 5. Start frontend (new terminal)
cd "../Supply Chain Resilience Website"
npm start  # Runs on port 3000
```

### Production Deployment

**Backend (Flask):**
```bash
# Use production WSGI server
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 backend.app:app
```

**Frontend (React):**
```bash
# Build for production
npm run build
# Serve with nginx or similar
```

**Docker (Optional):**
```dockerfile
# Backend Dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

---

## Security

### Current Implementation

⚠️ **Development Mode** - Not production-ready

**Missing Security Features:**
- ❌ No authentication
- ❌ No authorization
- ❌ No rate limiting
- ❌ No input sanitization
- ❌ No HTTPS enforcement
- ❌ No API keys

### Production Recommendations

✅ **Authentication:**
- Implement JWT tokens
- OAuth2 for enterprise SSO
- Session management

✅ **Authorization:**
- Role-based access control (RBAC)
- Company data isolation
- Admin vs. user permissions

✅ **API Security:**
- Rate limiting (e.g., 100 req/min)
- Input validation and sanitization
- SQL injection prevention (if using DB)
- XSS protection

✅ **Data Security:**
- Encrypt data at rest
- HTTPS/TLS for data in transit
- Secure file uploads
- Regular security audits

✅ **Infrastructure:**
- Firewall configuration
- DDoS protection
- Regular backups
- Monitoring and logging

---

## Performance Optimization

### Current Optimizations

✅ **Model Caching** - Load models once, reuse  
✅ **Batch Processing** - Process multiple nodes together  
✅ **GPU Acceleration** - CUDA support for training  
✅ **Lazy Loading** - Load data on demand  
✅ **Background Tasks** - Async training  

### Future Improvements

🔄 **Redis Caching** - Cache predictions  
🔄 **Database** - PostgreSQL for metadata  
🔄 **Message Queue** - RabbitMQ for tasks  
🔄 **Load Balancing** - Multiple backend instances  
🔄 **CDN** - Static asset delivery  

---

## Monitoring & Logging

### Current Logging

```python
# Backend logs
print(f"✅ Loaded model: {model_name}")
print(f"🔮 Making prediction...")
print(f"❌ Error: {error_message}")
```

### Production Monitoring

**Recommended Tools:**
- **Application:** Sentry, New Relic
- **Infrastructure:** Prometheus, Grafana
- **Logs:** ELK Stack (Elasticsearch, Logstash, Kibana)
- **Uptime:** Pingdom, UptimeRobot

**Key Metrics:**
- API response time
- Model inference latency
- Training job duration
- Error rates
- Resource utilization (CPU, GPU, memory)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | Apr 2026 | Initial release |

---

## Future Roadmap

### Phase 2 (Q3 2026)
- Real-time monitoring integration
- Multi-user support with authentication
- Advanced scenario templates
- Export to PDF reports

### Phase 3 (Q4 2026)
- Mobile app (React Native)
- ERP system integrations
- Automated retraining pipeline
- Advanced analytics dashboard

### Phase 4 (2027)
- Reinforcement learning for optimization
- Federated learning for privacy
- Blockchain for supply chain tracking
- IoT sensor integration

---

## Contributing

See `CONTRIBUTING.md` for development guidelines.

---

## License

See `LICENSE` file for details.

---

**Last Updated:** April 26, 2026  
**Maintained By:** Supply Chain Resilience Team
