import torch
from collections import Counter

labels = []
for i in range(100):
    data = torch.load(f'scenario_graphs_paper/scenario_{i:05d}.pt', weights_only=False)
    labels.extend(data.y[data.train_mask].tolist())

print('Label distribution (first 100 scenarios):', Counter(labels))
print('Total labeled nodes:', len(labels))
print('\nClass breakdown:')
for label, count in sorted(Counter(labels).items()):
    print(f'  Class {label}: {count} ({count/len(labels)*100:.1f}%)')
