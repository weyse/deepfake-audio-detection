import os, json, math
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ============================================================
# CONFIG — uses the best 5s model (layer 3, bottom)
# ============================================================
TARGET_DURATION = 5
TRESHOLD_LOW    = TARGET_DURATION / 3
LAYER_INDEX     = 3
HIDDEN_SIZES    = [512, 384, 256, 128]
THRESHOLD       = 0.6
BATCH_SIZE      = 32

CWD        = os.path.dirname(os.path.abspath(__file__))
REREC_DIR  = os.path.join(CWD, '..', '..', 'for-rerec', 'for-rerecorded')
MODEL_PATH = os.path.join(CWD, 'exp1_results', '5s', 'best_model.pth')
CACHE_PATH = os.path.join(CWD, 'exp1_results', '5s', 'rerec_test_cache.pt')
OUTPUT_DIR = os.path.join(CWD, 'exp1_results', '5s')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
print(f'REREC_DIR: {REREC_DIR}')

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
# Load audio files from folder structure
# ============================================================
def load_audio_paths(split):
    real_dir = os.path.join(REREC_DIR, split, 'real')
    fake_dir = os.path.join(REREC_DIR, split, 'fake')
    paths, labels = [], []
    for f in os.listdir(real_dir):
        if f.endswith('.wav'):
            paths.append(os.path.join(real_dir, f))
            labels.append(0)
    for f in os.listdir(fake_dir):
        if f.endswith('.wav'):
            paths.append(os.path.join(fake_dir, f))
            labels.append(1)
    return paths, labels

# ============================================================
# Slice and pad
# ============================================================
def process_audio(path, target_dur, low, sr=16000):
    max_samples = int(sr * target_dur)
    min_samples = int(sr * low)
    try:
        waveform, orig_sr = torchaudio.load(path)
        if orig_sr != sr:
            waveform = torchaudio.transforms.Resample(orig_sr, sr)(waveform)
        chunks = []
        if waveform.size(1) >= max_samples:
            n = math.ceil(waveform.size(1) / max_samples)
            for i in range(n):
                start = i * max_samples
                chunk = waveform[:, start:start + max_samples]
                if chunk.size(1) < min_samples:
                    continue
                if chunk.size(1) < max_samples:
                    chunk = torch.nn.functional.pad(chunk, (0, max_samples - chunk.size(1)))
                chunks.append(chunk)
        elif waveform.size(1) >= min_samples:
            chunk = torch.nn.functional.pad(waveform, (0, max_samples - waveform.size(1)))
            chunks.append(chunk)
        return chunks
    except Exception as e:
        print(f'  SKIP {path}: {e}')
        return []

# ============================================================
# Extract features
# ============================================================
if os.path.exists(CACHE_PATH):
    print('Loading rerec features from cache...')
    cache = torch.load(CACHE_PATH, weights_only=False)
    all_features = cache['features']
    all_labels   = cache['labels']
else:
    print('Loading wav2vec model...')
    bundle     = torchaudio.pipelines.WAV2VEC2_BASE
    feat_model = bundle.get_model().to(device)
    feat_model.eval()

    test_paths, test_labels = load_audio_paths('testing')
    print(f'Found {len(test_paths)} test files in for-rerec')

    all_features, all_labels = [], []
    for idx, (path, label) in enumerate(zip(test_paths, test_labels)):
        chunks = process_audio(path, TARGET_DURATION, TRESHOLD_LOW)
        for chunk in chunks:
            if chunk.ndim == 1:
                chunk = chunk.unsqueeze(0)
            chunk = chunk.to(device)
            with torch.inference_mode():
                hidden_states, _ = feat_model.extract_features(chunk)
            feat   = hidden_states[LAYER_INDEX].detach().cpu()
            pooled = feat.squeeze(0).mean(dim=0)
            all_features.append(pooled)
            all_labels.append(label)
        if (idx + 1) % 100 == 0:
            print(f'  [{idx+1}/{len(test_paths)}] done')
        torch.cuda.empty_cache()

    torch.save({'features': all_features, 'labels': all_labels}, CACHE_PATH)
    print(f'Cached to: {CACHE_PATH}')

print(f'Total test samples (after chunking): {len(all_features)}')
print(f'Real: {all_labels.count(0) if isinstance(all_labels, list) else (np.array(all_labels)==0).sum()} | Fake: {all_labels.count(1) if isinstance(all_labels, list) else (np.array(all_labels)==1).sum()}')

# ============================================================
# Load model and evaluate
# ============================================================
model = ResearchMLP(768, HIDDEN_SIZES, 1, 0.3).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()
print(f'Loaded model from: {MODEL_PATH}')

X = torch.stack([
    f if isinstance(f, torch.Tensor) else torch.tensor(f, dtype=torch.float32)
    for f in all_features
]).float()
y_true = np.array(all_labels)

all_preds = []
with torch.no_grad():
    for i in range(0, len(X), BATCH_SIZE):
        xb  = X[i:i+BATCH_SIZE].to(device)
        out  = model(xb)
        pred = (torch.sigmoid(out) > THRESHOLD).float().squeeze(1).cpu()
        all_preds.extend(pred.numpy())

y_pred = np.array(all_preds)

acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec  = recall_score(y_true, y_pred, zero_division=0)
f1   = f1_score(y_true, y_pred, zero_division=0)

print(f'\n=== CROSS-CONDITION RESULTS (for-rerec) ===')
print(f'Trained on : for-norm (5s, clean)')
print(f'Tested on  : for-rerec (voice channel simulation)')
print(f'N test     : {len(y_true)}')
print(f'Acc        : {acc*100:.2f}%')
print(f'Precision  : {prec*100:.2f}%')
print(f'Recall     : {rec*100:.2f}%')
print(f'F1 Score   : {f1*100:.2f}%')

results = {
    'cross_condition': 'for-norm -> for-rerec',
    'model': '5s_layer3_MLP4',
    'threshold': THRESHOLD,
    'n_test': int(len(y_true)),
    'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1
}
out_path = os.path.join(OUTPUT_DIR, 'rerec_results.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved to: {out_path}')
