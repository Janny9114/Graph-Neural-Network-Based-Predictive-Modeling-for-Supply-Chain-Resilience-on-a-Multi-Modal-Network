# API Documentation

## Supply Chain Resilience Platform - Backend API

**Base URL:** `http://localhost:5000/api`

---

## Table of Contents

1. [Health Check](#health-check)
2. [Graph Operations](#graph-operations)
3. [Prediction](#prediction)
4. [Training & Pipeline](#training--pipeline)
5. [Network Analysis](#network-analysis)
6. [Company Management](#company-management)
7. [Templates](#templates)

---

## Health Check

### GET `/health`

Check if the API is running and model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model": "GINE",
  "device": "cuda"
}
```

---

## Graph Operations

### GET `/graph`

Get graph data for visualization.

**Query Parameters:**
- `company_id` (optional): Company identifier

**Response:**
```json
{
  "nodes": [
    {
      "id": 0,
      "name": "Node_0",
      "tier": 0,
      "region": "North_America",
      "capacity": 1500.0,
      "reliability": 0.88,
      "risk_level": 0.35,
      "cost_factor": 0.75,
      "x": -74.006,
      "y": 40.7128
    }
  ],
  "edges": [
    {
      "source": 0,
      "target": 1,
      "weight": 0.6
    }
  ],
  "status": "success"
}
```

### GET `/overview-metrics`

Get overview metrics for the supply chain network.

**Query Parameters:**
- `company_id` (optional): Company identifier

**Response:**
```json
{
  "status": "success",
  "avg_resilience": 88.5,
  "avg_risk": 31.2,
  "avg_lead_time": 5.3,
  "network_density": 2.45
}
```

---

## Prediction

### POST `/predict`

Predict node resilience given a disruption scenario.

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
      "node_name": "Node_0",
      "label": 2,
      "label_name": "Normal",
      "probability": [0.1, 0.2, 0.7],
      "tier": 0,
      "region": "North_America"
    }
  ],
  "summary": {
    "failed": 45,
    "degraded": 78,
    "normal": 77,
    "total_nodes": 200
  },
  "model_info": {
    "model_name": "GINE",
    "company_id": "company_xyz"
  },
  "status": "success"
}
```

---

## Training & Pipeline

### POST `/upload-graph`

Upload company-specific graph files.

**Form Data:**
- `nodes`: CSV file with node data
- `edges`: CSV file with edge data
- `company_name`: Company name (string)

**Response:**
```json
{
  "status": "success",
  "company_id": "company_xyz_a1b2c3d4",
  "stats": {
    "num_nodes": 200,
    "num_edges": 450,
    "tiers": {"0": 50, "1": 60, "2": 70, "3": 20},
    "avg_capacity": 1500.0,
    "avg_risk": 0.35
  }
}
```

### POST `/run-complete-pipeline`

Run the complete training pipeline.

**Request Body:**
```json
{
  "num_scenarios": 2000,
  "use_default_data": true,
  "company_id": "optional_company_id"
}
```

**Response:**
```json
{
  "status": "started",
  "task_id": "a1b2c3d4e5f6",
  "message": "Pipeline started with 2000 scenarios. This may take 30-60 minutes."
}
```

### GET `/pipeline-status/<task_id>`

Get status of a running pipeline task.

**Response:**
```json
{
  "status": "running",
  "progress": 45,
  "message": "Training GNN models...",
  "stage": "training",
  "started_at": "2026-04-26T03:00:00",
  "updated_at": "2026-04-26T03:15:00"
}
```

### GET `/training-results`

Get training results from the latest run.

**Query Parameters:**
- `company_id` (optional): Company identifier

**Response:**
```json
{
  "status": "success",
  "results": [
    {
      "model": "GINE",
      "accuracy": 0.766,
      "precision": 0.758,
      "recall": 0.766,
      "f1": 0.762
    }
  ],
  "metadata": {
    "num_scenarios": 2000,
    "trained_at": "2026-04-26T03:00:00",
    "company_id": "company_xyz"
  }
}
```

---

## Network Analysis

### GET `/network-vulnerability`

Compute network vulnerability metrics.

**Query Parameters:**
- `top_n` (optional, default=10): Number of top nodes to return
- `company_id` (optional): Company identifier

**Response:**
```json
{
  "status": "success",
  "summary": {
    "num_nodes": 200,
    "num_edges": 450,
    "num_articulation_points": 15,
    "articulation_point_fraction": 0.075,
    "avg_betweenness": 0.005,
    "max_betweenness": 0.125,
    "centrality_method": "Eigenvector Centrality"
  },
  "betweenness_centrality": [...],
  "eigenvector_centrality": [...],
  "articulation_points": [...]
}
```

### GET `/network-topology`

Compute network topology metrics.

**Query Parameters:**
- `company_id` (optional): Company identifier

**Response:**
```json
{
  "status": "success",
  "metrics": {
    "density": {
      "value": 0.0225,
      "ideal": 0.15,
      "percentage": 15.0,
      "description": "Network interconnectedness"
    },
    "avg_path_length": {
      "value": 4.52,
      "description": "Average hops between nodes"
    },
    "clustering": {
      "value": 0.245,
      "ideal": 0.3,
      "percentage": 81.7,
      "description": "Local redundancy"
    }
  }
}
```

### GET `/cascading-failure`

Simulate cascading failures.

**Query Parameters:**
- `top_n` (optional, default=10): Number of nodes to simulate
- `company_id` (optional): Company identifier

**Response:**
```json
{
  "status": "success",
  "initial_state": {
    "num_nodes": 200,
    "num_edges": 450,
    "is_connected": true
  },
  "cascade_sequence": [
    {
      "node_id": 45,
      "node_name": "Node_45",
      "betweenness": 0.125,
      "nodes_disconnected": 1,
      "edges_lost": 8,
      "fragmentation_ratio": 0.005
    }
  ]
}
```

### POST `/trade-restriction-scenario`

Simulate trade restriction between regions.

**Request Body:**
```json
{
  "severity": 0.8,
  "region1": "China",
  "region2": "United_States"
}
```

**Response:**
```json
{
  "status": "success",
  "scenario": {
    "type": "trade_restriction",
    "severity": 0.8,
    "china_nodes_count": 45,
    "us_nodes_count": 38,
    "disrupted_edges_count": 67
  },
  "baseline": {
    "failed": 10,
    "degraded": 25,
    "normal": 165
  },
  "disrupted": {
    "failed": 45,
    "degraded": 78,
    "normal": 77
  },
  "impact": {
    "delta_failed": 35,
    "delta_degraded": 53,
    "delta_normal": -88,
    "nodes_worsened": 88
  }
}
```

---

## Company Management

### GET `/list-companies`

List all uploaded companies.

**Response:**
```json
{
  "status": "success",
  "companies": [
    "company_xyz_a1b2c3d4",
    "tech_corp_e5f6g7h8"
  ],
  "count": 2
}
```

### DELETE `/delete-company/<company_id>`

Delete all data for a specific company.

**Response:**
```json
{
  "status": "success",
  "message": "Successfully deleted all data for company company_xyz",
  "deleted_items": [
    "uploads/company_xyz/",
    "pipeline_output/company_xyz/"
  ]
}
```

---

## Templates

### GET `/download-template/<template_type>`

Download CSV template for nodes or edges.

**Path Parameters:**
- `template_type`: Either "nodes" or "edges"

**Response:**
- CSV file download

---

## Error Responses

All endpoints may return error responses in this format:

```json
{
  "status": "error",
  "message": "Error description",
  "traceback": "Detailed error trace (in debug mode)"
}
```

**Common HTTP Status Codes:**
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (resource doesn't exist)
- `500`: Internal Server Error

---

## Rate Limiting

Currently no rate limiting is implemented. For production deployment, consider adding rate limiting middleware.

---

## Authentication

Currently no authentication is required. For production deployment, implement JWT or OAuth2 authentication.

---

## CORS

CORS is enabled for all origins in development. For production, configure specific allowed origins in `backend/app.py`.

---

## WebSocket Support

Not currently implemented. For real-time progress updates, consider adding Socket.IO support.

---

## Version

**API Version:** 1.0.0  
**Last Updated:** April 26, 2026
