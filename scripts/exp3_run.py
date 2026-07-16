import os, json, subprocess, sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Install shap and lime if needed
for pkg in ['shap', 'lime']:
    try:
        __import__(pkg)
    except ImportError:
        print(f'Installing {pkg}...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '--quiet'])

import shap
from lime import lime_tabular
import joblib
import xgboost as xgb

# ============================================================
# CONFIG
# ============================================================
TARGET_DURATION         = 5
BEST_CLASSIFIER         = 'XGB'   # XGB has TreeExplainer (fast & exact)
SHAP_BACKGROUND_SAMPLES = 100
LIME_SAMPLE_IDX         = 0
LIME_NUM_FEATURES       = 20
NUM_LIME_SAMPLES        = 10
RUN_ATTENTION_VIZ       = False   # needs LSTM/Transformer model

CWD        = os.path.dirname(os.path.abspath(__file__))
EXP1_DIR   = os.path.join(CWD, 'exp1_results', f'{TARGET_DURATION}s')
EXP2_DIR   = os.path.join(CWD, 'exp2_results', f'{TARGET_DURATION}s')
OUTPUT_DIR = os.path.join(CWD, 'exp3_results', f'{TARGET_DURATION}s')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f'Classifier : {BEST_CLASSIFIER}')
print(f'Duration   : {TARGET_DURATION}s')
print(f'Output     : {OUTPUT_DIR}')

# ============================================================
# Step 1: Load pooled features
# ============================================================
print('\nLoading features...')
tr = torch.load(os.path.join(EXP1_DIR, 'pooled_train.pt'), weights_only=False)
vl = torch.load(os.path.join(EXP1_DIR, 'pooled_val.pt'),   weights_only=False)
te = torch.load(os.path.join(EXP1_DIR, 'pooled_test.pt'),  weights_only=False)

def to_numpy(x):
    if isinstance(x, np.ndarray): return x
    if isinstance(x, list):
        return np.array([f.numpy() if isinstance(f, torch.Tensor) else np.array(f) for f in x])
    return np.array(x)

train_X = to_numpy(tr['features'])
train_y = np.array(tr['labels'])
val_X   = to_numpy(vl['features'])
val_y   = np.array(vl['labels'])
test_X  = to_numpy(te['features'])
test_y  = np.array(te['labels'])

feature_names = [f'feat_{i}' for i in range(train_X.shape[1])]
print(f'train: {train_X.shape}, val: {val_X.shape}, test: {test_X.shape}')

# ============================================================
# Step 2: Load model
# ============================================================
print(f'\nLoading {BEST_CLASSIFIER} model...')
if BEST_CLASSIFIER == 'XGB':
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(EXP2_DIR, 'xgb_model.json'))
    model_type = 'tree'
    print('Loaded XGBoost model.')
elif BEST_CLASSIFIER == 'RF':
    model = joblib.load(os.path.join(EXP2_DIR, 'rf_model.pkl'))
    model_type = 'tree'
    print('Loaded Random Forest model.')

# ============================================================
# Step 3: SHAP Analysis
# ============================================================
print('\n--- SHAP Analysis ---')
np.random.seed(42)
bg_idx      = np.random.choice(len(train_X), size=min(SHAP_BACKGROUND_SAMPLES, len(train_X)), replace=False)
X_background = train_X[bg_idx]
X_explain    = test_X

print(f'Background: {X_background.shape} | Explain: {X_explain.shape}')
print(f'Creating SHAP explainer for {BEST_CLASSIFIER}...')

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_explain)

if isinstance(shap_values, list):
    shap_values = shap_values[1]  # class 1 = fake
shap_values = np.array(shap_values)
print(f'SHAP values shape: {shap_values.shape}')

# Plot 1: Top 20 feature importance bar chart
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top20_idx   = np.argsort(mean_abs_shap)[::-1][:20]
top20_vals  = mean_abs_shap[top20_idx]
top20_names = [feature_names[i] for i in top20_idx]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(20), top20_vals[::-1], color='steelblue')
ax.set_yticks(range(20))
ax.set_yticklabels(top20_names[::-1])
ax.set_xlabel('Mean |SHAP value|')
ax.set_title(f'Top 20 Most Important wav2vec Dimensions ({BEST_CLASSIFIER})')
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, 'shap_top20_importance.png')
plt.savefig(path, dpi=150)
plt.close()
print(f'Saved: shap_top20_importance.png')

# Plot 2: SHAP beeswarm
shap_explanation = shap.Explanation(
    values=shap_values,
    base_values=np.zeros(len(shap_values)),
    data=X_explain,
    feature_names=feature_names
)
plt.figure()
shap.plots.beeswarm(shap_explanation, max_display=20, show=False)
plt.title(f'SHAP Beeswarm: {BEST_CLASSIFIER}')
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, 'shap_beeswarm.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: shap_beeswarm.png')

# Plot 3: SHAP waterfall for single sample
true_label = 'FAKE' if int(test_y[LIME_SAMPLE_IDX]) == 1 else 'REAL'
single_shap = shap.Explanation(
    values=shap_values[LIME_SAMPLE_IDX],
    base_values=float(np.mean(shap_values)),
    data=X_explain[LIME_SAMPLE_IDX],
    feature_names=feature_names
)
plt.figure()
shap.plots.waterfall(single_shap, max_display=15, show=False)
plt.title(f'SHAP Waterfall: sample {LIME_SAMPLE_IDX} (true: {true_label})')
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, f'shap_waterfall_sample{LIME_SAMPLE_IDX}.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: shap_waterfall_sample{LIME_SAMPLE_IDX}.png')

# ============================================================
# Step 4: LIME Analysis
# ============================================================
print('\n--- LIME Analysis ---')

def predict_fn(X):
    return model.predict_proba(X)

lime_explainer = lime_tabular.LimeTabularExplainer(
    train_X,
    feature_names=feature_names,
    class_names=['real', 'fake'],
    mode='classification',
    random_state=42
)
print('LIME explainer ready.')

# Single sample explanation
sample     = test_X[LIME_SAMPLE_IDX]
true_label = 'FAKE' if int(test_y[LIME_SAMPLE_IDX]) == 1 else 'REAL'
print(f'Explaining sample {LIME_SAMPLE_IDX} (true: {true_label})...')

explanation = lime_explainer.explain_instance(
    sample, predict_fn,
    num_features=LIME_NUM_FEATURES,
    num_samples=1000
)

fig = explanation.as_pyplot_figure(label=1)
plt.title(f'LIME Explanation: sample {LIME_SAMPLE_IDX} (true: {true_label})')
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, f'lime_sample{LIME_SAMPLE_IDX}.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: lime_sample{LIME_SAMPLE_IDX}.png')

print('\nTop LIME features:')
for feat, weight in explanation.as_list(label=1)[:10]:
    print(f'  {feat}: {weight:.4f}')

# Aggregate LIME over multiple samples
print(f'\nAggregating LIME over {NUM_LIME_SAMPLES} test samples...')
all_lime_weights = {fn: [] for fn in feature_names}
for idx in range(min(NUM_LIME_SAMPLES, len(test_X))):
    exp = lime_explainer.explain_instance(
        test_X[idx], predict_fn,
        num_features=LIME_NUM_FEATURES, num_samples=500
    )
    for feat, weight in exp.as_list(label=1):
        feat_name = feat.split(' ')[0]
        if feat_name in all_lime_weights:
            all_lime_weights[feat_name].append(abs(weight))
    print(f'  Sample {idx+1}/{NUM_LIME_SAMPLES} done.')

lime_importance = {k: float(np.mean(v)) for k, v in all_lime_weights.items() if v}
top_lime = sorted(lime_importance.items(), key=lambda x: x[1], reverse=True)[:20]

# LIME aggregate bar chart
fig, ax = plt.subplots(figsize=(10, 6))
names, vals = zip(*top_lime)
ax.barh(range(len(names)), list(vals)[::-1], color='coral')
ax.set_yticks(range(len(names)))
ax.set_yticklabels(list(names)[::-1])
ax.set_xlabel('Mean |LIME weight|')
ax.set_title(f'Top 20 LIME Features ({BEST_CLASSIFIER}, {NUM_LIME_SAMPLES} samples)')
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, 'lime_aggregate_top20.png')
plt.savefig(path, dpi=150)
plt.close()
print(f'Saved: lime_aggregate_top20.png')

# ============================================================
# Step 5: Save XAI summary JSON
# ============================================================
top50_shap = [
    {'feature': feature_names[i], 'mean_abs_shap': float(mean_abs_shap[i])}
    for i in np.argsort(mean_abs_shap)[::-1][:50]
]

xai_summary = {
    'classifier'         : BEST_CLASSIFIER,
    'target_duration'    : TARGET_DURATION,
    'feature_dims'       : int(train_X.shape[1]),
    'shap_top50_features': top50_shap,
    'lime_top_features'  : [{'feature': n, 'mean_abs_lime_weight': v} for n, v in top_lime]
}

summary_path = os.path.join(OUTPUT_DIR, 'xai_summary.json')
with open(summary_path, 'w') as f:
    json.dump(xai_summary, f, indent=2)
print(f'\nXAI summary saved.')

print('\n=== ALL OUTPUTS ===')
for fname in sorted(os.listdir(OUTPUT_DIR)):
    print(f'  {fname}')

print('\nDone!')
