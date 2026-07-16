import os, time, json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ============================================================
# CONFIG
# ============================================================
TARGET_DURATION = 5
TRESHOLD_LOW    = TARGET_DURATION / 3
MODEL_SIZE      = 'base'
LAYER_INDEX     = 3    # Bottom layer
HIDDEN_SIZES    = [384, 192, 96, 48]
INPUT_SIZE      = 768
OUTPUT_SIZE     = 1
DROPOUT         = 0.3
EPOCHS          = 50
BATCH_SIZE      = 32
LEARNING_RATE   = 0.001
THRESHOLD       = 0.6

CWD            = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(CWD, 'checkpoint', 'contigency')
OUTPUT_DIR     = os.path.join(CWD, 'exp1_results', f'{TARGET_DURATION}s')
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ============================================================
# Load checkpoints
# ============================================================
print('Loading checkpoints...')
train_pt = torch.load(os.path.join(CHECKPOINT_DIR, 'complete_smote_train.pt'), weights_only=False)
val_pt   = torch.load(os.path.join(CHECKPOINT_DIR, 'complete_val.pt'), weights_only=False)
test_pt  = torch.load(os.path.join(CHECKPOINT_DIR, 'complete_test.pt'), weights_only=False)

print(f'Train: {len(train_pt["features"])} samples')
print(f'Val:   {len(val_pt["features"])} samples')
print(f'Test:  {len(test_pt["features"])} samples')

# ============================================================
# Dataset
# ============================================================
class CameraReadyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict):
        self.features = torch.stack([
            f if isinstance(f, torch.Tensor) else torch.tensor(f, dtype=torch.float32)
            for f in data_dict['features']
        ]).to(torch.float32)
        self.labels = torch.tensor(list(data_dict['labels']), dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

dataset_train = CameraReadyDataset(train_pt)
dataset_val   = CameraReadyDataset(val_pt)
dataset_test  = CameraReadyDataset(test_pt)

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

model = ResearchMLP(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE, DROPOUT).to(device)
print(f'\nModel: {INPUT_SIZE} -> {HIDDEN_SIZES} -> {OUTPUT_SIZE}')
print(f'Threshold: {THRESHOLD}')

# ============================================================
# Training
# ============================================================
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_loss    = float('inf')
patience         = 5
patience_counter = 0
start_time       = time.perf_counter()

for epoch in range(EPOCHS):
    model.train()
    loader_train = DataLoader(dataset_train, BATCH_SIZE, shuffle=True)
    loader_val   = DataLoader(dataset_val,   BATCH_SIZE, shuffle=False)

    train_correct, train_total = 0, 0
    for features, labels in loader_train:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss    = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        preds = (torch.sigmoid(outputs) > THRESHOLD).float()
        train_correct += (preds == labels.unsqueeze(1)).sum().item()
        train_total   += labels.size(0)

    train_acc = train_correct / train_total

    model.eval()
    total_val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for features, labels in loader_val:
            features, labels = features.to(device), labels.to(device)
            outputs  = model(features)
            val_loss = criterion(outputs, labels.unsqueeze(1))
            preds    = (torch.sigmoid(outputs) > THRESHOLD).float()
            total_val_loss += val_loss.item()
            val_correct    += (preds == labels.unsqueeze(1)).sum().item()
            val_total      += labels.size(0)

    avg_val_loss = total_val_loss / len(loader_val)
    val_acc      = val_correct / val_total

    print(f'Epoch {epoch+1:>2}/{EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {avg_val_loss:.4f}')

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
        print(f'  -> Best model saved.')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

# ============================================================
# Test Evaluation
# ============================================================
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth')))
model.eval()

loader_test = DataLoader(dataset_test, BATCH_SIZE, shuffle=False)
all_preds, all_labels = [], []

with torch.no_grad():
    for features, labels in loader_test:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        preds   = (torch.sigmoid(outputs) > THRESHOLD).float()
        all_preds.extend(preds.squeeze(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

test_acc       = accuracy_score(all_labels, all_preds)
test_precision = precision_score(all_labels, all_preds, zero_division=0)
test_recall    = recall_score(all_labels, all_preds, zero_division=0)
test_f1        = f1_score(all_labels, all_preds, zero_division=0)
runtime        = time.perf_counter() - start_time

print(f'\n=== RESULTS ===')
print(f'Train Acc  : {train_acc:.4f}')
print(f'Val Acc    : {val_acc:.4f}')
print(f'Test Acc   : {test_acc:.4f}  ({test_acc*100:.2f}%)')
print(f'Precision  : {test_precision:.4f}  ({test_precision*100:.2f}%)')
print(f'Recall     : {test_recall:.4f}  ({test_recall*100:.2f}%)')
print(f'F1 Score   : {test_f1:.4f}  ({test_f1*100:.2f}%)')
print(f'Runtime    : {runtime:.1f}s')

# ============================================================
# Save results
# ============================================================
results = {
    'target_duration_seconds'  : TARGET_DURATION,
    'treshold_low_seconds'     : TRESHOLD_LOW,
    'model_size'               : MODEL_SIZE,
    'layer_index'              : LAYER_INDEX,
    'hidden_sizes'             : HIDDEN_SIZES,
    'threshold'                : THRESHOLD,
    'epochs_run'               : EPOCHS,
    'batch_size'               : BATCH_SIZE,
    'learning_rate'            : LEARNING_RATE,
    'runtime_seconds'          : runtime,
    'train_accuracy'           : train_acc,
    'validation_accuracy'      : val_acc,
    'test_accuracy'            : test_acc,
    'test_precision'           : test_precision,
    'test_recall'              : test_recall,
    'test_f1'                  : test_f1,
    'train_samples_after_smote': len(dataset_train),
    'val_samples'              : len(dataset_val),
    'test_samples'             : len(dataset_test),
}

results_path = os.path.join(OUTPUT_DIR, 'model_results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f'Results saved to: {results_path}')
