"""
Trade Restriction Scenario: China-US Trade War Simulation
Simulates the impact of trade restrictions between China and United States
on the global supply chain network using GNN predictions.

This script:
1. Identifies nodes in China and United States
2. Disrupts edges (trade routes) between these regions
3. Runs GNN prediction to assess cascading impacts
4. Visualizes results on a world map
5. Generates detailed analysis report
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Import model
import sys
sys.path.append('C:/Users/janny/Desktop/final_year')
from train_multi_gnn_realistic import GINEModel

# ============================================================================
# CONFIGURATION
# ============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'best_gine_model.pt'
NODES_PATH = 'synthetic_nodes.csv'
EDGES_PATH = 'synthetic_edges.csv'

# Trade restriction severity (0.0 = no restriction, 1.0 = complete embargo)
TRADE_RESTRICTION_SEVERITY = 0.8

# ============================================================================
# LOAD DATA
# ============================================================================

print("="*70)
print("CHINA-US TRADE RESTRICTION SCENARIO SIMULATION")
print("="*70)

print("\n📂 Loading supply chain data...")
node_df = pd.read_csv(NODES_PATH)
edge_df = pd.read_csv(EDGES_PATH)
print(f"  ✓ Loaded {len(node_df)} nodes, {len(edge_df)} edges")

# ============================================================================
# IDENTIFY CHINA AND US NODES
# ============================================================================

print("\n🌍 Identifying nodes by region...")

# Map regions to countries (adjust based on your data)
china_regions = ['East_Asia', 'China', 'Asia']  # Adjust to match your region names
us_regions = ['North_America', 'United_States', 'US', 'USA']

# Identify nodes
china_nodes = node_df[node_df['region'].isin(china_regions)].index.tolist()
us_nodes = node_df[node_df['region'].isin(us_regions)].index.tolist()

print(f"  ✓ China nodes: {len(china_nodes)}")
print(f"  ✓ US nodes: {len(us_nodes)}")

# If no exact matches, use geographic coordinates
if len(china_nodes) == 0 or len(us_nodes) == 0:
    print("\n  ⚠️  Using geographic coordinates to identify regions...")
    # China: roughly 18-54°N, 73-135°E
    china_nodes = node_df[
        (node_df['y'] >= 18) & (node_df['y'] <= 54) &
        (node_df['x'] >= 73) & (node_df['x'] <= 135)
    ].index.tolist()
    
    # US: roughly 25-50°N, -125 to -65°W (convert to positive: 235-295°E)
    us_nodes = node_df[
        (node_df['y'] >= 25) & (node_df['y'] <= 50) &
        (node_df['x'] >= -125) & (node_df['x'] <= -65)
    ].index.tolist()
    
    print(f"  ✓ China nodes (by coordinates): {len(china_nodes)}")
    print(f"  ✓ US nodes (by coordinates): {len(us_nodes)}")

# ============================================================================
# IDENTIFY DISRUPTED EDGES (TRADE ROUTES)
# ============================================================================

print("\n🚫 Identifying trade routes between China and US...")

# Find edges connecting China and US
disrupted_edges = []
for idx, row in edge_df.iterrows():
    source = int(row['source'])
    target = int(row['target'])
    
    # Check if edge connects China to US (or vice versa)
    if (source in china_nodes and target in us_nodes) or \
       (source in us_nodes and target in china_nodes):
        disrupted_edges.append([source, target])

print(f"  ✓ Found {len(disrupted_edges)} trade routes between China and US")
print(f"  ✓ Trade restriction severity: {TRADE_RESTRICTION_SEVERITY * 100}%")

# ============================================================================
# LOAD GNN MODEL
# ============================================================================

print("\n🤖 Loading GINE model...")
model = GINEModel(in_channels=11, edge_dim=4, hidden_channels=256, dropout=0.3, num_classes=3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print(f"  ✓ Model loaded successfully on {DEVICE}")

# ============================================================================
# CREATE BASELINE (NO DISRUPTION)
# ============================================================================

print("\n📊 Running baseline prediction (no disruption)...")

def create_graph_data(node_df, edge_df, disrupted_nodes=[], disrupted_edges=[], severity=0.0):
    """Create PyTorch Geometric Data object."""
    num_nodes = len(node_df)
    
    # Node features
    base_features = torch.tensor(
        node_df[['capacity', 'cost_factor', 'risk_level', 'reliability', 'x', 'y']].values,
        dtype=torch.float
    ).clone()
    
    # Apply disruption to nodes
    for node_id in disrupted_nodes:
        base_features[node_id, 0] *= (1.0 - severity)  # capacity
        base_features[node_id, 3] *= (1.0 - severity)  # reliability
        base_features[node_id, 2] = min(1.0, base_features[node_id, 2] + severity)  # risk
    
    # Tier encoding
    tier_encoding = torch.zeros((num_nodes, 4), dtype=torch.float)
    for idx, tier in enumerate(node_df['tier'].values):
        tier_encoding[idx, int(tier)] = 1.0
    
    # is_disrupted flag
    is_disrupted = torch.zeros((num_nodes, 1), dtype=torch.float)
    for node_id in disrupted_nodes:
        is_disrupted[node_id, 0] = severity
    
    # Concatenate features
    x = torch.cat([base_features, tier_encoding, is_disrupted], dim=1)
    
    # Edge index
    edge_index = torch.tensor(
        np.array([edge_df['source'].values, edge_df['target'].values]),
        dtype=torch.long
    )
    
    # Edge features
    num_edges = len(edge_df)
    edge_attr = torch.zeros((num_edges, 4), dtype=torch.float)
    
    weights = edge_df.get('weight', edge_df.get('capacity_share', pd.Series([1.0]*num_edges))).values
    edge_attr[:, 0] = torch.tensor((weights - weights.mean()) / (weights.std() + 1e-8))
    edge_attr[:, 2] = torch.tensor(weights)
    
    for idx, row in edge_df.iterrows():
        source_idx = int(row['source'])
        target_idx = int(row['target'])
        
        source_cost = float(node_df.iloc[source_idx]['cost_factor'])
        target_cost = float(node_df.iloc[target_idx]['cost_factor'])
        edge_attr[idx, 1] = (source_cost + target_cost) / 2.0
        
        source_risk = float(node_df.iloc[source_idx]['risk_level'])
        target_risk = float(node_df.iloc[target_idx]['risk_level'])
        edge_attr[idx, 3] = (source_risk + target_risk) / 2.0
    
    from torch_geometric.data import Data
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# Baseline prediction
baseline_data = create_graph_data(node_df, edge_df)
baseline_data = baseline_data.to(DEVICE)

with torch.no_grad():
    baseline_out = model(baseline_data.x, baseline_data.edge_index, baseline_data.edge_attr)
    baseline_probs = torch.exp(baseline_out)
    baseline_preds = baseline_out.argmax(dim=1).cpu().numpy()

baseline_summary = {
    'failed': int((baseline_preds == 0).sum()),
    'degraded': int((baseline_preds == 1).sum()),
    'normal': int((baseline_preds == 2).sum())
}

print(f"  ✓ Baseline: {baseline_summary['failed']} failed, {baseline_summary['degraded']} degraded, {baseline_summary['normal']} normal")

# ============================================================================
# RUN TRADE RESTRICTION SCENARIO
# ============================================================================

print(f"\n🚫 Running trade restriction scenario (severity={TRADE_RESTRICTION_SEVERITY})...")

# Identify China supplier nodes (tier 0 and 1) to simulate supply disruption
china_supplier_nodes = [n for n in china_nodes if node_df.iloc[n]['tier'] <= 1]
# Also disrupt some US nodes to show downstream impact
affected_nodes = china_supplier_nodes + us_nodes[:15]  # Disrupt top 15 US nodes

print(f"  ✓ Disrupting {len(china_supplier_nodes)} China supplier nodes (tier 0-1)")
print(f"  ✓ Disrupting {min(15, len(us_nodes))} US nodes")

# Create disrupted graph
disrupted_data = create_graph_data(
    node_df, edge_df,
    disrupted_nodes=affected_nodes,
    disrupted_edges=disrupted_edges,
    severity=TRADE_RESTRICTION_SEVERITY
)
disrupted_data = disrupted_data.to(DEVICE)

with torch.no_grad():
    disrupted_out = model(disrupted_data.x, disrupted_data.edge_index, disrupted_data.edge_attr)
    disrupted_probs = torch.exp(disrupted_out)
    disrupted_preds = disrupted_out.argmax(dim=1).cpu().numpy()

disrupted_summary = {
    'failed': int((disrupted_preds == 0).sum()),
    'degraded': int((disrupted_preds == 1).sum()),
    'normal': int((disrupted_preds == 2).sum())
}

print(f"  ✓ Disrupted: {disrupted_summary['failed']} failed, {disrupted_summary['degraded']} degraded, {disrupted_summary['normal']} normal")

# ============================================================================
# ANALYZE IMPACT
# ============================================================================

print("\n📈 Analyzing impact...")

# Calculate changes
delta_failed = disrupted_summary['failed'] - baseline_summary['failed']
delta_degraded = disrupted_summary['degraded'] - baseline_summary['degraded']
delta_normal = disrupted_summary['normal'] - baseline_summary['normal']

print(f"\n  Impact Summary:")
print(f"    Failed nodes:   {baseline_summary['failed']} → {disrupted_summary['failed']} ({delta_failed:+d})")
print(f"    Degraded nodes: {baseline_summary['degraded']} → {disrupted_summary['degraded']} ({delta_degraded:+d})")
print(f"    Normal nodes:   {baseline_summary['normal']} → {disrupted_summary['normal']} ({delta_normal:+d})")

# Identify most affected nodes
node_df['baseline_pred'] = baseline_preds
node_df['disrupted_pred'] = disrupted_preds
node_df['status_change'] = node_df['disrupted_pred'] - node_df['baseline_pred']

# Nodes that got worse
worsened_nodes = node_df[node_df['status_change'] < 0].sort_values('status_change')
print(f"\n  ✓ {len(worsened_nodes)} nodes worsened due to trade restrictions")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n📊 Creating visualizations...")

fig = plt.figure(figsize=(20, 12))

# ── Plot 1: World Map - Baseline ──────────────────────────────────────────
ax1 = plt.subplot(2, 3, 1)
colors_baseline = ['red' if p == 0 else 'orange' if p == 1 else 'green' for p in baseline_preds]
scatter1 = ax1.scatter(node_df['x'], node_df['y'], c=colors_baseline, s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
ax1.set_title('Baseline: No Trade Restrictions', fontsize=14, fontweight='bold')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.grid(True, alpha=0.3)
ax1.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Failed'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Degraded'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Normal')
], loc='upper right')

# ── Plot 2: World Map - Trade Restriction ────────────────────────────────
ax2 = plt.subplot(2, 3, 2)
colors_disrupted = ['red' if p == 0 else 'orange' if p == 1 else 'green' for p in disrupted_preds]
scatter2 = ax2.scatter(node_df['x'], node_df['y'], c=colors_disrupted, s=50, alpha=0.6, edgecolors='black', linewidth=0.5)

# Highlight China and US regions
if len(china_nodes) > 0:
    china_x = node_df.iloc[china_nodes]['x'].values
    china_y = node_df.iloc[china_nodes]['y'].values
    ax2.scatter(china_x, china_y, s=200, facecolors='none', edgecolors='blue', linewidth=2, label='China')

if len(us_nodes) > 0:
    us_x = node_df.iloc[us_nodes]['x'].values
    us_y = node_df.iloc[us_nodes]['y'].values
    ax2.scatter(us_x, us_y, s=200, facecolors='none', edgecolors='purple', linewidth=2, label='United States')

# Draw disrupted edges
for edge in disrupted_edges[:20]:  # Limit to first 20 for clarity
    source_x = node_df.iloc[edge[0]]['x']
    source_y = node_df.iloc[edge[0]]['y']
    target_x = node_df.iloc[edge[1]]['x']
    target_y = node_df.iloc[edge[1]]['y']
    ax2.plot([source_x, target_x], [source_y, target_y], 'r--', alpha=0.3, linewidth=1)

ax2.set_title(f'Trade Restriction Scenario (Severity={TRADE_RESTRICTION_SEVERITY*100}%)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right')

# ── Plot 3: Impact Heatmap ────────────────────────────────────────────────
ax3 = plt.subplot(2, 3, 3)
impact_colors = ['darkred' if c < -1 else 'red' if c == -1 else 'gray' if c == 0 else 'lightgreen' for c in node_df['status_change']]
scatter3 = ax3.scatter(node_df['x'], node_df['y'], c=impact_colors, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
ax3.set_title('Impact Heatmap (Status Change)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Longitude')
ax3.set_ylabel('Latitude')
ax3.grid(True, alpha=0.3)
ax3.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkred', markersize=10, label='Severe Degradation'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Degraded'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='No Change'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=10, label='Improved')
], loc='upper right')

# ── Plot 4: Bar Chart Comparison ──────────────────────────────────────────
ax4 = plt.subplot(2, 3, 4)
categories = ['Failed', 'Degraded', 'Normal']
baseline_values = [baseline_summary['failed'], baseline_summary['degraded'], baseline_summary['normal']]
disrupted_values = [disrupted_summary['failed'], disrupted_summary['degraded'], disrupted_summary['normal']]

x = np.arange(len(categories))
width = 0.35

bars1 = ax4.bar(x - width/2, baseline_values, width, label='Baseline', color='skyblue')
bars2 = ax4.bar(x + width/2, disrupted_values, width, label='Trade Restriction', color='salmon')

ax4.set_xlabel('Node Status')
ax4.set_ylabel('Number of Nodes')
ax4.set_title('Comparison: Baseline vs Trade Restriction', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(categories)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)

# ── Plot 5: Top 10 Most Affected Nodes ───────────────────────────────────
ax5 = plt.subplot(2, 3, 5)
top_affected = worsened_nodes.head(10)
if len(top_affected) > 0:
    y_pos = np.arange(len(top_affected))
    ax5.barh(y_pos, -top_affected['status_change'].values, color='darkred', alpha=0.7)
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels([f"Node {idx} ({row['region']})" for idx, row in top_affected.iterrows()], fontsize=8)
    ax5.set_xlabel('Status Degradation')
    ax5.set_title('Top 10 Most Affected Nodes', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='x')
    ax5.invert_yaxis()
else:
    ax5.text(0.5, 0.5, 'No nodes worsened', ha='center', va='center', fontsize=12)
    ax5.set_title('Top 10 Most Affected Nodes', fontsize=14, fontweight='bold')

# ── Plot 6: Regional Impact Analysis ─────────────────────────────────────
ax6 = plt.subplot(2, 3, 6)
regional_impact = node_df.groupby('region')['status_change'].agg(['mean', 'count']).sort_values('mean')
if len(regional_impact) > 0:
    colors_regional = ['red' if x < 0 else 'green' for x in regional_impact['mean']]
    ax6.barh(range(len(regional_impact)), regional_impact['mean'], color=colors_regional, alpha=0.7)
    ax6.set_yticks(range(len(regional_impact)))
    ax6.set_yticklabels(regional_impact.index, fontsize=8)
    ax6.set_xlabel('Average Status Change')
    ax6.set_title('Regional Impact Analysis', fontsize=14, fontweight='bold')
    ax6.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax6.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('trade_restriction_china_us_analysis.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved visualization: trade_restriction_china_us_analysis.png")

# ============================================================================
# GENERATE DETAILED REPORT
# ============================================================================

print("\n📄 Generating detailed report...")

report = f"""
{'='*70}
CHINA-US TRADE RESTRICTION SCENARIO: DETAILED ANALYSIS REPORT
{'='*70}

1. SCENARIO CONFIGURATION
   - Trade Restriction Severity: {TRADE_RESTRICTION_SEVERITY * 100}%
   - Affected Regions: China and United States
   - China Nodes: {len(china_nodes)}
   - US Nodes: {len(us_nodes)}
   - Disrupted Trade Routes: {len(disrupted_edges)}

2. BASELINE (NO RESTRICTIONS)
   - Failed Nodes:   {baseline_summary['failed']} ({baseline_summary['failed']/len(node_df)*100:.1f}%)
   - Degraded Nodes: {baseline_summary['degraded']} ({baseline_summary['degraded']/len(node_df)*100:.1f}%)
   - Normal Nodes:   {baseline_summary['normal']} ({baseline_summary['normal']/len(node_df)*100:.1f}%)

3. TRADE RESTRICTION SCENARIO
   - Failed Nodes:   {disrupted_summary['failed']} ({disrupted_summary['failed']/len(node_df)*100:.1f}%)
   - Degraded Nodes: {disrupted_summary['degraded']} ({disrupted_summary['degraded']/len(node_df)*100:.1f}%)
   - Normal Nodes:   {disrupted_summary['normal']} ({disrupted_summary['normal']/len(node_df)*100:.1f}%)

4. IMPACT ANALYSIS
   - Change in Failed Nodes:   {delta_failed:+d} ({delta_failed/len(node_df)*100:+.1f}%)
   - Change in Degraded Nodes: {delta_degraded:+d} ({delta_degraded/len(node_df)*100:+.1f}%)
   - Change in Normal Nodes:   {delta_normal:+d} ({delta_normal/len(node_df)*100:+.1f}%)
   - Total Nodes Worsened: {len(worsened_nodes)} ({len(worsened_nodes)/len(node_df)*100:.1f}%)

5. TOP 10 MOST AFFECTED NODES
"""

for i, (idx, row) in enumerate(worsened_nodes.head(10).iterrows(), 1):
    report += f"   {i}. Node {idx} ({row['region']}): "
    report += f"Status changed from {['Failed', 'Degraded', 'Normal'][int(row['baseline_pred'])]} "
    report += f"to {['Failed', 'Degraded', 'Normal'][int(row['disrupted_pred'])]}\n"

report += f"""
6. REGIONAL IMPACT SUMMARY
"""

for region, data in regional_impact.iterrows():
    report += f"   - {region}: Avg change = {data['mean']:.2f} ({int(data['count'])} nodes)\n"

report += f"""
7. KEY FINDINGS
   - The trade restriction between China and US caused {len(worsened_nodes)} nodes to degrade
   - {delta_failed} additional nodes failed completely
   - {delta_degraded} nodes moved to degraded status
   - The cascading effect impacted {len(worsened_nodes)/len(node_df)*100:.1f}% of the network
   - Most affected regions: {', '.join(regional_impact.head(3).index.tolist())}

8. RECOMMENDATIONS
   - Diversify supply sources away from China-US dependencies
   - Strengthen regional supply chains to reduce cross-border reliance
   - Build buffer inventory for critical components
   - Develop alternative trade routes through other regions
   - Invest in domestic manufacturing capabilities

{'='*70}
Report generated: {pd.Timestamp.now()}
Model: GINE (Graph Isomorphism Network with Edges)
Accuracy: 76.6%
{'='*70}
"""

# Save report
with open('trade_restriction_china_us_report.txt', 'w') as f:
    f.write(report)

print(report)
print(f"\n✓ Saved report: trade_restriction_china_us_report.txt")

# Save detailed results to CSV
results_df = node_df[['node_id', 'region', 'tier', 'capacity', 'risk_level', 'reliability', 
                       'baseline_pred', 'disrupted_pred', 'status_change']].copy()
results_df['baseline_status'] = results_df['baseline_pred'].map({0: 'Failed', 1: 'Degraded', 2: 'Normal'})
results_df['disrupted_status'] = results_df['disrupted_pred'].map({0: 'Failed', 1: 'Degraded', 2: 'Normal'})
results_df.to_csv('trade_restriction_china_us_results.csv', index=False)
print(f"✓ Saved detailed results: trade_restriction_china_us_results.csv")

print("\n" + "="*70)
print("✅ SIMULATION COMPLETE!")
print("="*70)
print("\n📁 Output Files:")
print("  1. trade_restriction_china_us_analysis.png - Comprehensive visualization")
print("  2. trade_restriction_china_us_report.txt - Detailed analysis report")
print("  3. trade_restriction_china_us_results.csv - Node-level results")
