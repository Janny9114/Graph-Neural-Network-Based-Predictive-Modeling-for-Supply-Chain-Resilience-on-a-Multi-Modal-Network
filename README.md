User Guide
Supply Chain Resilience Platform
Version: 1.0.0
Last Updated: April 26, 2026

Table of Contents
Introduction
Getting Started
Dashboard Overview
Features
Workflows
Troubleshooting
FAQ
Introduction
The Supply Chain Resilience Platform is an AI-powered tool that helps organizations:

Predict supply chain disruptions using Graph Neural Networks (GNNs)
Analyze network vulnerabilities and critical nodes
Simulate various disruption scenarios (natural disasters, trade restrictions, etc.)
Optimize supply chain resilience strategies
Key Capabilities
✅ Real-time Predictions - Instant disruption impact analysis
✅ Custom Networks - Upload your own supply chain data
✅ Multiple Models - 6 GNN architectures + 4 ML baselines
✅ Interactive Visualizations - 3D network maps and analytics
✅ Scenario Simulation - Test "what-if" scenarios

Getting Started
Prerequisites
Web Browser: Chrome, Firefox, or Edge (latest version)
Internet Connection: Required for initial setup
Screen Resolution: 1920x1080 or higher recommended
First Time Setup
Access the Platform

Open your browser and navigate to: http://localhost:3000
You should see the welcome screen
Choose Your Path

Option A: Use default demo data (recommended for first-time users)
Option B: Upload your own supply chain network
Wait for Model Loading

The platform loads pre-trained AI models (~30 seconds)
You'll see a loading indicator
Start Exploring!

Once loaded, you can access all features
Dashboard Overview
Main Navigation
The platform has 6 main sections:

📊 Dashboard - Overview metrics and quick insights
🗺️ Network Map - Interactive 3D visualization
🔮 Predictions - Run disruption simulations
📈 Analytics - Detailed network analysis
⚙️ Training - Train custom models
📁 Data Management - Upload/manage data
Dashboard Metrics
Top Cards:

Average Resilience - Overall network health (0-100%)
Average Risk - Network vulnerability level (0-100%)
Network Density - Connectivity measure
Critical Nodes - Number of high-risk nodes
Features
1. Network Visualization
Access: Dashboard → Network Map

What it does:

Displays your supply chain as an interactive 3D graph
Nodes = Suppliers, factories, warehouses, retailers
Edges = Supply relationships
How to use:

Click and drag to rotate the view
Scroll to zoom in/out
Click on nodes to see details
Use filters to highlight specific tiers or regions
Node Colors:

🔵 Blue - Tier 0 (Raw material suppliers)
🟢 Green - Tier 1 (Component manufacturers)
🟡 Yellow - Tier 2 (Assembly plants)
🔴 Red - Tier 3 (Distribution/retail)
2. Disruption Prediction
Access: Dashboard → Predictions

What it does:

Predicts how disruptions propagate through your network
Shows which nodes will fail, degrade, or remain operational
How to use:

Step 1: Select Disruption Type
Node Disruption - Facility failures (fire, flood, strike)
Edge Disruption - Transportation issues (port closure, route blockage)
Regional Disruption - Area-wide events (earthquake, pandemic)
Step 2: Configure Scenario
Select Nodes/Edges - Click on map or enter IDs
Set Severity - 0% (minor) to 100% (catastrophic)
Adjust Buffer - Inventory cushion (0-100%)
Step 3: Run Prediction
Click "Predict Impact"
Wait 2-5 seconds for results
View predictions on map and table
Step 4: Analyze Results
Failed Nodes (Red) - Complete operational failure
Degraded Nodes (Yellow) - Reduced capacity
Normal Nodes (Green) - Unaffected
Example Scenario:

Disruption: Earthquake in East Asia
Affected Nodes: 5 suppliers in China
Severity: 80%
Result: 45 nodes failed, 78 degraded, 77 normal
3. Network Vulnerability Analysis
Access: Dashboard → Analytics → Vulnerability

What it does:

Identifies critical nodes whose failure would cause maximum damage
Calculates centrality metrics
Key Metrics:

Betweenness Centrality

Measures how often a node lies on shortest paths
High value = bottleneck/critical bridge
Action: Increase redundancy for high-betweenness nodes
Eigenvector Centrality

Measures influence based on connections to other important nodes
High value = hub with many important connections
Action: Protect high-influence hubs
Articulation Points

Nodes whose removal disconnects the network
Critical! Single points of failure
Action: Create backup routes immediately
4. Cascading Failure Simulation
Access: Dashboard → Analytics → Cascading Failure

What it does:

Simulates sequential node failures
Shows how network fragments over time
How to interpret:

Fragmentation Ratio - % of network disconnected
Largest Component - Size of main connected cluster
Number of Components - How many isolated groups
Use Case: "If we lose our top 5 critical nodes sequentially, how badly is the network damaged?"

5. Trade Restriction Scenario
Access: Dashboard → Predictions → Trade Restrictions

What it does:

Simulates geopolitical trade barriers
Shows impact of blocked trade routes between regions
Example:

Scenario: China-US Trade War
Severity: 80% tariffs/restrictions
Result:
- 67 trade routes disrupted
- 88 nodes worsened
- 35 additional failures
Regions Available:

North America
Europe
East Asia
South Asia
Middle East
Africa
South America
Oceania
6. Custom Model Training
Access: Dashboard → Training → Run Pipeline

What it does:

Trains AI models on your specific supply chain
Compares 6 GNN architectures + 4 ML baselines
Takes 30-60 minutes
Steps:

Upload Your Data

Nodes CSV: supplier/facility information
Edges CSV: supply relationships
Configure Training

Number of scenarios: 2000 (recommended)
Training epochs: 300 (default)
Start Training

Click "Start Training"
Monitor progress bar
Check back in 30-60 minutes
View Results

Model comparison table
Best model automatically selected
Download training report
Model Comparison:

Model	Accuracy	F1 Score	Speed
GINE	76.6%	76.2%	Fast
GAT	75.8%	75.4%	Medium
GraphSAGE	74.2%	73.8%	Fast
GCN	73.5%	73.1%	Very Fast
Workflows
Workflow 1: Quick Risk Assessment
Goal: Get a quick overview of supply chain health

Steps:

Go to Dashboard
Check top 4 metrics
View Network Map for visual overview
Check Vulnerability Analysis for critical nodes
Time: 5 minutes
Workflow 2: Disaster Impact Simulation
Goal: Predict impact of a specific disaster

Steps:

Go to Predictions
Select "Node Disruption"
Choose affected region on map
Set severity (e.g., 80% for major earthquake)
Click "Predict Impact"
Analyze results:
How many nodes failed?
Which critical suppliers affected?
What's the recovery strategy?
Time: 10 minutes
Workflow 3: Identify Critical Suppliers
Goal: Find which suppliers are most critical

Steps:

Go to Analytics → Vulnerability
Sort by Betweenness Centrality (descending)
Note top 10 nodes
Check if they're Articulation Points
For each critical node:
Document backup suppliers
Increase inventory buffers
Establish alternative routes
Time: 20 minutes
Workflow 4: Custom Network Training
Goal: Train models on your own supply chain

Steps:

Prepare your data (see Data Format below)
Go to Data Management → Upload
Upload nodes.csv and edges.csv
Validate data (automatic)
Go to Training → Run Pipeline
Configure: 2000 scenarios, 300 epochs
Start training
Wait 30-60 minutes
Review results and download report
Time: 1-2 hours (mostly automated)
Data Format
Nodes CSV Format
Required Columns:

node_id,name,tier,capacity,cost_factor,risk_level,reliability,latitude,longitude,region
0,Supplier_A,0,1500,0.75,0.35,0.88,40.7128,-74.0060,North_America
1,Factory_X,1,2000,0.82,0.28,0.91,35.6762,139.6503,Asia
Column Descriptions:

node_id: Unique identifier (0, 1, 2, ...)
name: Human-readable name
tier: Supply chain level (0=raw materials, 3=retail)
capacity: Production/storage capacity (units/day)
cost_factor: Relative cost (0.0-1.0)
risk_level: Vulnerability score (0.0-1.0)
reliability: Historical uptime (0.0-1.0)
latitude/longitude: Geographic coordinates
region: Geographic region name
Edges CSV Format
Required Columns:

source,target,capacity_share,lead_time,cost
0,1,0.6,7,1500
1,2,0.8,5,800
Column Descriptions:

source: Source node ID
target: Target node ID
capacity_share: % of capacity allocated (0.0-1.0)
lead_time: Shipping time (days)
cost: Transportation cost ($)
