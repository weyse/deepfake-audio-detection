import os, time, json, math, random
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import to_MLP_Utility as utl
import importlib; importlib.reload(utl)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print('XGBoost not found, skipping. Install with: pip install xgboost')

# ============================================================
# CONFIG
# ============================================================
TARGET_DURATION = 5
MODEL_SIZE      = 'base'
LAYER_INDEX     = 3
EPOCHS          = 50
BATCH_SIZE      = 32
LEARNING_RATE   = 0.001

RUN_MLP  = False  # already done
RUN_SVM  = False  # too slow
RUN_RF   = True
RUN_XGB  = True
RUN_LSTM = False  # needs frame-level features
RUN_ATTN = False  # needs frame-level features

CWD        = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR  = os.path.join(CWD, 'exp1_results', f'{TARGET_DURATION}s', 'cache')
OUTPUT_DIR = os.path.join(CWD, 'exp2_results', f'{TARGET_DURATION}s')
EXP1_PT_DIR = os.path.join(CWD, 'exp1_results', f'{TARGET_DURATION}s')
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ============================================================
# Step 1: Load pooled features from exp1 cache
# ============================================================
pooled_train_path = os.path.join(EXP1_PT_DIR, 'pooled_train.pt')
pooled_val_path   = os.path.join(EXP1_PT_DIR, 'pooled_val.pt')
pooled_test_path  = os.path.join(EXP1_PT_DIR, 'pooled_test.pt')

if os.path.exists(pooled_train_path):
    print('Loading pre-computed pooled features...')
    tr = torch.load(pooled_train_path, weights_only=False)
    vl = torch.load(pooled_val_path,   weights_only=False)
    te = torch.load(pooled_test_path,  weights_only=False)
    train_X = tr['features']
    train_y = tr['labels']
    val_X   = vl['features']
    val_y   = vl['labels']
    test_X  = te['features']
    test_y  = te['labels']
else:
    print('Pooled features not found, converting from exp1 cache...')
    def load_cache(split):
        path = os.path.join(CACHE_DIR, f'{split}.pt')
        data = torch.load(path, weights_only=False)
        X = np.array([f.numpy() if isinstance(f, torch.Tensor) else np.array(f)
                      for f in data['features']])
        y = np.array(data['labels'])
        return X, y

    train_X, train_y = load_cache('train')
    val_X,   val_y   = load_cache('val')
    test_X,  test_y  = load_cache('test')

    # Save as pooled_*.pt for future use
    torch.save({'features': train_X, 'labels': train_y}, pooled_train_path)
    torch.save({'features': val_X,   'labels': val_y},   pooled_val_path)
    torch.save({'features': test_X,  'labels': test_y},  pooled_test_path)
    print('Saved pooled feature files.')

# Ensure numpy arrays
if isinstance(train_X, list):
    train_X = np.array([f.numpy() if isinstance(f, torch.Tensor) else np.array(f) for f in train_X])
    train_y = np.array(train_y)
    val_X   = np.array([f.numpy() if isinstance(f, torch.Tensor) else np.array(f) for f in val_X])
    val_y   = np.array(val_y)
    test_X  = np.array([f.numpy() if isinstance(f, torch.Tensor) else np.array(f) for f in test_X])
    test_y  = np.array(test_y)

print(f'train_X: {train_X.shape}, val_X: {val_X.shape}, test_X: {test_X.shape}')

# ============================================================
# Step 2: SMOTE + Scaler
# ============================================================
sm = SMOTE(random_state=42)
train_X_bal, train_y_bal = sm.fit_resample(train_X, train_y)
print(f'After SMOTE: {train_X_bal.shape}')

scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X_bal)
val_X_scaled   = scaler.transform(val_X)
test_X_scaled  = scaler.transform(test_X)

all_results = {}

# ============================================================
# Classifier 1: MLP
# ============================================================
if RUN_MLP:
    print('\n=== MLP_3layer ===')

    class ResearchMLP(nn.Module):
        def __init__(self, input_size, hidden_sizes, output_size, dropout):
            super().__init__()
            layers, prev = [], input_size
            for i, h in enumerate(hidden_sizes):
                layers += [nn.Linear(prev, h), nn.ReLU()]
                if i < len(hidden_sizes) - 1:
                    layers.append(nn.Dropout(dropout))
                prev = h
            layers.append(nn.Linear(prev, output_size))
            self.network = nn.Sequential(*layers)
        def forward(self, x): return self.network(x)

    X_t = torch.FloatTensor(train_X_bal).to(device)
    y_t = torch.FloatTensor(train_y_bal).to(device)
    X_v = torch.FloatTensor(val_X).to(device)
    y_v = torch.FloatTensor(val_y).to(device)
    X_te = torch.FloatTensor(test_X).to(device)
    y_te = torch.FloatTensor(test_y).to(device)

    ds_train = TensorDataset(X_t, y_t)
    ds_val   = TensorDataset(X_v, y_v)

    mlp = ResearchMLP(768, [384, 192, 96], 1, 0.3).to(device)
    criterion  = nn.BCEWithLogitsLoss()
    optimizer  = optim.Adam(mlp.parameters(), lr=LEARNING_RATE)
    best_val   = float('inf')
    patience_c = 0
    start = time.perf_counter()

    for epoch in range(EPOCHS):
        mlp.train()
        for xb, yb in DataLoader(ds_train, BATCH_SIZE, shuffle=True):
            optimizer.zero_grad()
            loss = criterion(mlp(xb), yb.unsqueeze(1))
            loss.backward(); optimizer.step()

        mlp.eval()
        with torch.no_grad():
            val_loss = criterion(mlp(X_v), y_v.unsqueeze(1)).item()
            val_preds = (torch.sigmoid(mlp(X_v)) > 0.5).float()
            val_acc = (val_preds == y_v.unsqueeze(1)).float().mean().item()

        if val_loss < best_val:
            best_val = val_loss; patience_c = 0
            torch.save(mlp.state_dict(), os.path.join(OUTPUT_DIR, 'mlp_model.pth'))
        else:
            patience_c += 1
            if patience_c >= 5:
                print(f'  Early stopping at epoch {epoch+1}'); break

        if (epoch+1) % 5 == 0:
            print(f'  Epoch {epoch+1}/{EPOCHS} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}')

    mlp.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'mlp_model.pth'), weights_only=True))
    mlp.eval()
    with torch.no_grad():
        test_preds = (torch.sigmoid(mlp(X_te)) > 0.5).float()
        test_acc = (test_preds == y_te.unsqueeze(1)).float().mean().item()
    runtime = time.perf_counter() - start
    all_results['MLP'] = {'val_accuracy': val_acc, 'test_accuracy': test_acc, 'runtime_seconds': runtime}
    print(f'  Val: {val_acc:.4f} | Test: {test_acc:.4f} | Time: {runtime:.1f}s')

# ============================================================
# Classifier 2: SVM
# ============================================================
if RUN_SVM:
    print('\n=== SVM (RBF kernel) ===')
    start = time.perf_counter()
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    svm.fit(train_X_scaled, train_y_bal)
    val_acc  = accuracy_score(val_y,  svm.predict(val_X_scaled))
    test_acc = accuracy_score(test_y, svm.predict(test_X_scaled))
    runtime  = time.perf_counter() - start
    import joblib; joblib.dump(svm, os.path.join(OUTPUT_DIR, 'svm_model.pkl'))
    all_results['SVM'] = {'val_accuracy': val_acc, 'test_accuracy': test_acc, 'runtime_seconds': runtime}
    print(f'  Val: {val_acc:.4f} | Test: {test_acc:.4f} | Time: {runtime:.1f}s')

# ============================================================
# Classifier 3: Random Forest
# ============================================================
if RUN_RF:
    print('\n=== Random Forest ===')
    start = time.perf_counter()
    rf = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, random_state=42)
    rf.fit(train_X_bal, train_y_bal)
    val_acc  = accuracy_score(val_y,  rf.predict(val_X))
    test_acc = accuracy_score(test_y, rf.predict(test_X))
    runtime  = time.perf_counter() - start
    import joblib; joblib.dump(rf, os.path.join(OUTPUT_DIR, 'rf_model.pkl'))
    all_results['RandomForest'] = {'val_accuracy': val_acc, 'test_accuracy': test_acc, 'runtime_seconds': runtime}
    print(f'  Val: {val_acc:.4f} | Test: {test_acc:.4f} | Time: {runtime:.1f}s')

# ============================================================
# Classifier 4: XGBoost
# ============================================================
if RUN_XGB and XGB_AVAILABLE:
    print('\n=== XGBoost ===')
    start = time.perf_counter()
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        eval_metric='logloss', n_jobs=-1, random_state=42
    )
    xgb_model.fit(train_X_bal, train_y_bal, eval_set=[(val_X, val_y)], verbose=50)
    val_acc  = accuracy_score(val_y,  xgb_model.predict(val_X))
    test_acc = accuracy_score(test_y, xgb_model.predict(test_X))
    runtime  = time.perf_counter() - start
    xgb_model.save_model(os.path.join(OUTPUT_DIR, 'xgb_model.json'))
    all_results['XGBoost'] = {'val_accuracy': val_acc, 'test_accuracy': test_acc, 'runtime_seconds': runtime}
    print(f'  Val: {val_acc:.4f} | Test: {test_acc:.4f} | Time: {runtime:.1f}s')

# ============================================================
# Classifier 5: LSTM (frame-level features)
# ============================================================
if RUN_LSTM:
    print('\n=== LSTM (re-extracting frame-level features) ===')
    lstm_cache_train = os.path.join(OUTPUT_DIR, 'lstm_features_train.pt')
    lstm_cache_val   = os.path.join(OUTPUT_DIR, 'lstm_features_val.pt')
    lstm_cache_test  = os.path.join(OUTPUT_DIR, 'lstm_features_test.pt')

    if os.path.exists(lstm_cache_train):
        print('  Loading LSTM features from cache...')
        lt = torch.load(lstm_cache_train, weights_only=False)
        lv = torch.load(lstm_cache_val,   weights_only=False)
        lte = torch.load(lstm_cache_test,  weights_only=False)
        lstm_train_feats, lstm_train_labels = lt['features'], lt['labels']
        lstm_val_feats,   lstm_val_labels   = lv['features'], lv['labels']
        lstm_test_feats,  lstm_test_labels  = lte['features'], lte['labels']
    else:
        print('  Extracting frame-level features...')
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        feat_model = bundle.get_model().to(device)
        feat_model.eval()

        def flatten_waveform(s):
            while isinstance(s, list): s = s[0]
            return s

        def extract_frame_level(waveforms, labels, name, batch=500):
            all_f, all_l = [], list(labels)
            total = len(waveforms)
            for start in range(0, total, batch):
                end = min(start+batch, total)
                for idx, s in enumerate(waveforms[start:end], start=start+1):
                    wf = flatten_waveform(s)
                    if isinstance(wf, list): wf = torch.tensor(wf)
                    if wf.ndim == 1: wf = wf.unsqueeze(0)
                    wf = wf.to(device)
                    with torch.inference_mode():
                        hidden_states, _ = feat_model.extract_features(wf)
                    feat = hidden_states[LAYER_INDEX].detach().cpu().squeeze(0)  # [T, 768]
                    all_f.append(feat)
                    if idx % 1000 == 0: print(f'  [{idx}/{total}] OK')
                torch.cuda.empty_cache()
            return all_f, all_l

        # Load waveforms from exp1 cache (pooled cache has labels but not waveforms)
        # We'll use the checkpoint data
        chk_train = torch.load(os.path.join(CWD, 'checkpoint', 'contigency', 'complete_smote_train.pt'), weights_only=False)
        chk_val   = torch.load(os.path.join(CWD, 'checkpoint', 'contigency', 'complete_val.pt'),         weights_only=False)
        chk_test  = torch.load(os.path.join(CWD, 'checkpoint', 'contigency', 'complete_test.pt'),        weights_only=False)

        # These are already pooled 768-dim. For LSTM we need to expand back.
        # Skip LSTM since we only have pooled features saved, not raw waveforms in checkpoint.
        print('  NOTE: Raw waveforms not cached. Skipping LSTM to save time.')
        print('  (To enable LSTM, re-run exp1_run.py and save raw waveforms separately.)')
        RUN_LSTM = False
        RUN_ATTN = False

# ============================================================
# Classifier 6: Transformer (Attention)
# ============================================================
if RUN_ATTN:
    print('\n=== Transformer (Attention) ===')
    # Same frame-level features as LSTM needed
    print('  Skipping: requires frame-level features (same as LSTM).')

# ============================================================
# Summary
# ============================================================
print('\n=== EXP 2 SUMMARY ===')
print(f'{"Classifier":<20} {"Val Acc":>10} {"Test Acc":>10} {"Runtime (s)":>12}')
print('-' * 55)
for name, res in all_results.items():
    print(f'{name:<20} {res["val_accuracy"]:>10.4f} {res["test_accuracy"]:>10.4f} {res["runtime_seconds"]:>12.1f}')

results_path = os.path.join(OUTPUT_DIR, 'all_results.json')
with open(results_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f'\nSaved to {results_path}')
