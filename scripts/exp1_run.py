import os, math, time, json, random
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from imblearn.over_sampling import SMOTE
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import to_MLP_Utility as utl
import importlib; importlib.reload(utl)

# ============================================================
# CONFIG
# ============================================================
TARGET_DURATION = 5
TRESHOLD_LOW    = TARGET_DURATION / 3
MODEL_SIZE      = 'base'
LAYER_INDEX     = 3    # Bottom layer (early acoustic features)
HIDDEN_SIZES    = [384, 192, 96, 48]
INPUT_SIZE      = 768
DROPOUT         = 0.3
EPOCHS          = 50
BATCH_SIZE      = 32
LEARNING_RATE   = 0.001
EXTRACT_BATCH   = 2000  # process this many samples at a time to save RAM

CWD          = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(CWD, 'datasets_separated')
OUTPUT_DIR   = os.path.join(CWD, 'exp1_results', f'{TARGET_DURATION}s')
CACHE_DIR    = os.path.join(CWD, 'exp1_results', f'{TARGET_DURATION}s', 'cache')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

print(f'CWD        : {CWD}')
print(f'TARGET_DURATION : {TARGET_DURATION}s  |  TRESHOLD_LOW: {TRESHOLD_LOW:.2f}s')
print(f'GPU: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')

# ============================================================
# Step 1: Load paths from CSV
# ============================================================
def load_all_paths(split, label, datasets_dir):
    name   = f'{label}_{split}'
    folder = os.path.join(datasets_dir, name)
    outliers = pd.read_csv(os.path.join(folder, 'outliers.csv'))
    below    = pd.read_csv(os.path.join(folder, f'{name}_below_treshold.csv'))
    for df in [outliers, below]:
        if 'Unnamed: 0' in df.columns:
            df.drop('Unnamed: 0', axis=1, inplace=True)
    combined = pd.concat([outliers, below], ignore_index=True)
    combined = combined.dropna(subset=['file path', 'duration'])
    combined['duration'] = combined['duration'].astype(float)
    return combined

df_real_train = load_all_paths('training',   'real', DATASETS_DIR)
df_fake_train = load_all_paths('training',   'fake', DATASETS_DIR)
df_real_val   = load_all_paths('validation', 'real', DATASETS_DIR)
df_fake_val   = load_all_paths('validation', 'fake', DATASETS_DIR)
df_real_test  = load_all_paths('testing',    'real', DATASETS_DIR)
df_fake_test  = load_all_paths('testing',    'fake', DATASETS_DIR)

print('Loaded paths:')
for name, df in [('real_train', df_real_train), ('fake_train', df_fake_train),
                 ('real_val',   df_real_val),   ('fake_val',   df_fake_val),
                 ('real_test',  df_real_test),  ('fake_test',  df_fake_test)]:
    print(f'  {name}: {len(df)} files')

# ============================================================
# Step 2+3: Re-separate + Slice and Pad
# ============================================================
def re_separate(df, target_duration, low):
    need_slice = df[df['duration'] >= target_duration].reset_index(drop=True)
    need_pad   = df[(df['duration'] >= low) & (df['duration'] < target_duration)].reset_index(drop=True)
    discarded  = df[df['duration'] < low]
    return need_slice, need_pad, len(discarded)

def slice_and_pad(need_slice_df, need_pad_df, target_duration, low, sample_rate=16000):
    max_samples = int(sample_rate * target_duration)
    min_samples = int(sample_rate * low)
    chunks = []
    for _, row in need_slice_df.iterrows():
        try:
            waveform, sr = torchaudio.load(row['file path'])
            if sr != sample_rate:
                waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
            n = math.ceil(waveform.size(1) / max_samples)
            for i in range(n):
                start = i * max_samples
                chunk = waveform[:, start:start + max_samples]
                if chunk.size(1) < min_samples:
                    continue
                if chunk.size(1) < max_samples:
                    chunk = torch.nn.functional.pad(chunk, (0, max_samples - chunk.size(1)))
                chunks.append(chunk)
        except Exception as e:
            print(f'  SKIP slice {row["file path"]}: {e}')
    for _, row in need_pad_df.iterrows():
        try:
            waveform, sr = torchaudio.load(row['file path'])
            if sr != sample_rate:
                waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
            if waveform.size(1) > max_samples:
                waveform = waveform[:, :max_samples]
            chunk = torch.nn.functional.pad(waveform, (0, max_samples - waveform.size(1)))
            chunks.append(chunk)
        except Exception as e:
            print(f'  SKIP pad {row["file path"]}: {e}')
    return chunks

print('\nSlicing and padding audio...')
real_train_chunks = slice_and_pad(*re_separate(df_real_train, TARGET_DURATION, TRESHOLD_LOW)[:2], TARGET_DURATION, TRESHOLD_LOW)
fake_train_chunks = slice_and_pad(*re_separate(df_fake_train, TARGET_DURATION, TRESHOLD_LOW)[:2], TARGET_DURATION, TRESHOLD_LOW)
real_val_chunks   = slice_and_pad(*re_separate(df_real_val,   TARGET_DURATION, TRESHOLD_LOW)[:2], TARGET_DURATION, TRESHOLD_LOW)
fake_val_chunks   = slice_and_pad(*re_separate(df_fake_val,   TARGET_DURATION, TRESHOLD_LOW)[:2], TARGET_DURATION, TRESHOLD_LOW)
real_test_chunks  = slice_and_pad(*re_separate(df_real_test,  TARGET_DURATION, TRESHOLD_LOW)[:2], TARGET_DURATION, TRESHOLD_LOW)
fake_test_chunks  = slice_and_pad(*re_separate(df_fake_test,  TARGET_DURATION, TRESHOLD_LOW)[:2], TARGET_DURATION, TRESHOLD_LOW)

print(f'real_train: {len(real_train_chunks)} | fake_train: {len(fake_train_chunks)}')
print(f'real_val:   {len(real_val_chunks)}   | fake_val:   {len(fake_val_chunks)}')
print(f'real_test:  {len(real_test_chunks)}  | fake_test:  {len(fake_test_chunks)}')

# ============================================================
# Step 4: Build Labeled Sequences
# ============================================================
def build_labeled(real_chunks, fake_chunks, shuffle=True):
    dataset = [(w, 0) for w in real_chunks] + [(w, 1) for w in fake_chunks]
    if shuffle:
        random.seed(42)
        random.shuffle(dataset)
    return dataset

train_seq = build_labeled(real_train_chunks, fake_train_chunks, shuffle=True)
val_seq   = build_labeled(real_val_chunks,   fake_val_chunks,   shuffle=False)
test_seq  = build_labeled(real_test_chunks,  fake_test_chunks,  shuffle=False)

train_waveforms, train_labels = zip(*train_seq)
val_waveforms,   val_labels   = zip(*val_seq)
test_waveforms,  test_labels  = zip(*test_seq)

print(f'\ntrain: {len(train_waveforms)}, val: {len(val_waveforms)}, test: {len(test_waveforms)}')

# Free memory after building sequences
del real_train_chunks, fake_train_chunks
del real_val_chunks, fake_val_chunks
del real_test_chunks, fake_test_chunks

# ============================================================
# Step 5: Batched Feature Extraction (saves RAM)
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bundle = torchaudio.pipelines.WAV2VEC2_BASE
model  = bundle.get_model().to(device)
model.eval()
print(f'\nUsing model: WAV2VEC2_BASE on {device}')

def flatten_waveform(sample):
    while isinstance(sample, list):
        sample = sample[0]
    return sample

def extract_and_pool_batched(waveforms, labels, split_name, batch_size=EXTRACT_BATCH):
    cache_path = os.path.join(CACHE_DIR, f'{split_name}.pt')
    if os.path.exists(cache_path):
        print(f'  [{split_name}] Loading from cache...')
        data = torch.load(cache_path)
        return data['features'], data['labels']

    all_features = []
    all_labels   = list(labels)
    total        = len(waveforms)
    print(f'  [{split_name}] Extracting {total} samples in batches of {batch_size}...')

    for start in range(0, total, batch_size):
        end   = min(start + batch_size, total)
        batch = waveforms[start:end]
        batch_features = []
        for idx, sample in enumerate(batch, start=start+1):
            wf = flatten_waveform(sample)
            if isinstance(wf, list):
                wf = torch.tensor(wf)
            if wf.ndim == 1:
                wf = wf.unsqueeze(0)
            wf = wf.to(device)
            with torch.inference_mode():
                hidden_states, _ = model.extract_features(wf)
            feature = hidden_states[LAYER_INDEX].detach().cpu()
            pooled  = feature.squeeze(0).mean(dim=0)  # [768]
            batch_features.append(pooled)
            if idx % 500 == 0:
                print(f'    [{idx}/{total}] OK')
        all_features.extend(batch_features)
        print(f'  Batch {start}-{end} done, total so far: {len(all_features)}')
        torch.cuda.empty_cache()

    torch.save({'features': all_features, 'labels': all_labels}, cache_path)
    print(f'  [{split_name}] Saved to cache.')
    return all_features, all_labels

print('\nExtracting features...')
train_features, train_labels_list = extract_and_pool_batched(train_waveforms, train_labels, 'train')
val_features,   val_labels_list   = extract_and_pool_batched(val_waveforms,   val_labels,   'val')
test_features,  test_labels_list  = extract_and_pool_batched(test_waveforms,  test_labels,  'test')

print(f'\nFeature shape: {train_features[0].shape}')

# ============================================================
# Step 7: SMOTE + Save checkpoint
# ============================================================
print('\nApplying SMOTE...')
X_train = np.array([f.numpy() for f in train_features])
y_train = np.array(train_labels_list)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
print(f'After SMOTE: {X_resampled.shape}, labels: {np.bincount(y_resampled)}')

checkpoint_dir = os.path.join(CWD, 'checkpoint', 'contigency')
os.makedirs(checkpoint_dir, exist_ok=True)

torch.save({'features': [torch.tensor(x, dtype=torch.float32) for x in X_resampled],
            'labels':   list(y_resampled)},
           os.path.join(checkpoint_dir, 'complete_smote_train.pt'))

torch.save({'features': [torch.tensor(f.numpy(), dtype=torch.float32) for f in val_features],
            'labels':   val_labels_list},
           os.path.join(checkpoint_dir, 'complete_val.pt'))

torch.save({'features': [torch.tensor(f.numpy(), dtype=torch.float32) for f in test_features],
            'labels':   test_labels_list},
           os.path.join(checkpoint_dir, 'complete_test.pt'))

print('Checkpoints saved to:', checkpoint_dir)
print('Done!')
