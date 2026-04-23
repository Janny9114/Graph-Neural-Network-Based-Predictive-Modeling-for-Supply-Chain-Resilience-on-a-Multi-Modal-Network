import torch
import numpy as np

# Load a scenario
data = torch.load('C:/Users/janny/Desktop/final_year/scenario_graphs_edge_disruptions/scenario_00000.pt', weights_only=False)

# Check labels
labels = data.y.numpy()
print('Unique labels:', np.unique(labels))
print('\nLabel counts:')
for label in np.unique(labels):
    count = (labels == label).sum()
    print(f'  Label {label}: {count} nodes ({count/len(labels)*100:.1f}%)')

print(f'\nAny -1 labels? {-1 in labels}')
print(f'Total nodes: {len(labels)}')

# Check train_mask
if hasattr(data, 'train_mask'):
    train_mask = data.train_mask.numpy()
    print(f'\nTrain mask:')
    print(f'  Labeled nodes (train_mask=True): {train_mask.sum()}')
    print(f'  Unlabeled nodes (train_mask=False): {(~train_mask).sum()}')
    
    # Check labels of unlabeled nodes
    unlabeled_labels = labels[~train_mask]
    print(f'\nLabels of unlabeled nodes (train_mask=False):')
    print(f'  Unique labels: {np.unique(unlabeled_labels)}')
