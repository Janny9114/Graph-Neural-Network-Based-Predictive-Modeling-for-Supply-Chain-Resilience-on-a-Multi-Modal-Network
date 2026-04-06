# 1. Load graph structure
data = torch.load('supply_chain_graph.pt')

# 2. Load resilience labels
labels_df = pd.read_csv('node_resilience_labels.csv')
data.y = torch.tensor(labels_df['resilient'].values)

# 3. Train GAT model
model = GAT(in_channels=10, hidden_channels=64, out_channels=2)
# ... training loop ...

# 4. Predict resilience for new scenarios
predictions = model(data.x, data.edge_index)
