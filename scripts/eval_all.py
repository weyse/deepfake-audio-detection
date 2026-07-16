import os, json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

CWD = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Model
# ============================================================
class ResearchMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout):
        super().__init__()
        layers = []
        prev = input_size
        for i, h in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if i < len(hidden_sizes) - 1:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ============================================================
# Config per duration
# ============================================================
CONFIGS = {
    '2s':  {'hidden': [384, 192, 96],     'threshold': 0.5, 'layer': 11},
    '4s':  {'hidden': [384, 192, 96],     'threshold': 0.5, 'layer': 11},
    '6s':  {'hidden': [384, 192, 96],     'threshold': 0.5, 'layer': 11},
    '8s':  {'hidden': [384, 192, 96],     'threshold': 0.5, 'layer': 11},
    '5s':  {'hidden': [512, 384, 256, 128], 'threshold': 0.6, 'layer': 3},
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
print()

all_results = {}

for dur, cfg in CONFIGS.items():
    cache_path = os.path.join(CWD, 'exp1_results', dur, 'cache', 'test.pt')
    model_path = os.path.join(CWD, 'exp1_results', dur, 'best_model.pth')

    if not os.path.exists(cache_path):
        print(f'[{dur}] Cache not found, skipping.')
        continue
    if not os.path.exists(model_path):
        print(f'[{dur}] Model not found, skipping.')
        continue

    # Load test features
    data = torch.load(cache_path, weights_only=False)
    features = data['features']
    labels   = data['labels']

    X = torch.stack([
        f if isinstance(f, torch.Tensor) else torch.tensor(f, dtype=torch.float32)
        for f in features
    ]).float()
    y = torch.tensor(list(labels), dtype=torch.float32)

    # Load model
    model = ResearchMLP(768, cfg['hidden'], 1, 0.3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Evaluate
    all_preds, all_labels = [], []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            xb = X[i:i+64].to(device)
            yb = y[i:i+64]
            out  = model(xb)
            pred = (torch.sigmoid(out) > cfg['threshold']).float().squeeze(1).cpu()
            all_preds.extend(pred.numpy())
            all_labels.extend(yb.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec  = recall_score(all_labels, all_preds, zero_division=0)
    f1   = f1_score(all_labels, all_preds, zero_division=0)

    print(f'[{dur}] Layer={cfg["layer"]} | Threshold={cfg["threshold"]} | n={len(all_labels)}')
    print(f'       Acc={acc*100:.2f}%  Prec={prec*100:.2f}%  Recall={rec*100:.2f}%  F1={f1*100:.2f}%')
    print()

    all_results[dur] = {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
        'n_test': len(all_labels), 'layer': cfg['layer'], 'threshold': cfg['threshold']
    }

# Save
out_path = os.path.join(CWD, 'eval_all_results.json')
with open(out_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f'Saved to: {out_path}')
