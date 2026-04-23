aimport torch
import numpy as np

# Check multiple scenarios to see label distribution
print("Checking first 10 scenarios for label distribution...\n")

for i in range(10):
    data = torch.load(f'C:/Users/janny/Desktop/final_year/scenario_graphs_edge_disruptions/scenario_{i:05d}.pt', weights_only=False)
    labels = data.y.numpy()
    unique_labels = np.unique(labels)
    
    print(f"Scenario {i:05d}:")
    print(f"  Unique labels: {unique_labels}")
    for label in unique_labels:
        count = (labels == label).sum()
        print(f"    Label {label}: {count} nodes ({count/len(labels)*100:.1f}%)")
    print()
