# Deepfake Audio Detection with Wav2Vec 2.0

This repository contains our project on detecting deepfake (AI-generated) speech using Wav2Vec 2.0 embeddings combined with an MLP classifier. It was developed as part of a group assignment (Kelompok 3), trained and evaluated on the Fake-or-Real (FoR) dataset.

The core idea is simple: instead of hand-crafting audio features, we use Wav2Vec 2.0's pretrained speech representations as input to a classifier, then test how well that setup holds up across different conditions.

## Approach

Audio clips are sliced/padded to a fixed duration, then passed through `torchaudio`'s pretrained WAV2VEC2_BASE model. We pool the hidden states from a chosen transformer layer (mean over time) to get a 768-dim embedding per clip, balance the training set with SMOTE, and feed that into a classifier.

## Experiments

- **Duration variation** (`exp1`) – tests clip lengths of 2s, 4s, 5s, 6s, and 8s to see which works best. 5s ended up being the strongest setup.
- **Classifier variation** (`exp2`) – compares MLP, Random Forest, and XGBoost on top of the same wav2vec embeddings (SVM and LSTM/attention were tried but skipped — SVM was too slow at this data size, and the LSTM/attention variants need frame-level features we didn't end up caching).
- **Explainable AI** (`exp3`) – uses SHAP and LIME on the best-performing classifier to see which embedding dimensions actually drive the fake/real decision, instead of just reporting accuracy.
- **Cross-condition robustness** (`rerec_test.py`) – takes the best 5s model (trained on clean audio) and tests it on re-recorded audio (audio played back and re-recorded through a different channel), to check how much performance drops outside the training distribution.

## Dataset

We used the Fake-or-Real (FoR) dataset, specifically the normalized split for training/evaluation and the re-recorded split for the robustness test. *(link the source here)*

The raw dataset isn't included in this repo since it's too large for GitHub — see `data/README.md` for how to get it.

## Repository Structure

```
deepfake-audio-detection/
├── README.md
├── requirements.txt
├── data/
│   └── README.md              # dataset download instructions
├── notebooks/
│   ├── preprocess.ipynb
│   ├── exp1_duration_variation.ipynb
│   ├── exp2_classifier_variation.ipynb
│   └── exp3_explainable_ai.ipynb
├── scripts/
│   ├── exp1_train.py
│   ├── exp1_run.py
│   ├── exp2_run.py
│   ├── exp3_run.py
│   ├── eval_all.py
│   ├── rerec_test.py
│   └── to_MLP_Utility.py
├── results/
│   └── eval_all_results.json
└── docs/
    ├── Kelompok3_Manuscript.pdf
    └── Kelompok3_PPT.pdf
```

## Setup

```bash
git clone https://github.com/weyse/deepfake-audio-detection.git
cd deepfake-audio-detection
pip install -r requirements.txt
```

## Running it

Start with preprocessing:
```bash
jupyter notebook notebooks/preprocess.ipynb
```

Train the model:
```bash
python scripts/exp1_train.py
```

Run the experiments:
```bash
python scripts/exp1_run.py
python scripts/exp2_run.py
python scripts/exp3_run.py
```

Then evaluate everything together:
```bash
python scripts/eval_all.py
```
This writes results to `results/eval_all_results.json`.

## Results

Test set performance across the different clip durations (from `eval_all_results.json`):

| Duration | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| 2s | 71.5% | 84.8% | 43.2% | 57.2% |
| 4s | 78.6% | 93.3% | 52.6% | 67.3% |
| 6s | 85.8% | 88.3% | 48.3% | 62.5% |
| 8s | 87.3% | 81.7% | 31.0% | 45.0% |
| **5s** | **88.3%** | **97.6%** | **65.4%** | **78.3%** |

The 5-second clips (using layer 3 embeddings and a 4-layer MLP) gave the best overall result — highest accuracy, precision, and F1. Precision is consistently strong across all durations, but recall is the weak point, meaning the model misses a fair number of fake samples rather than falsely flagging real ones.

On the cross-condition test (`rerec_test.py`), the 5s model was evaluated on re-recorded audio to check generalization outside the clean training distribution — see `rerec_results.json` for those numbers once you run it.

## Docs

The full write-up and slides are in `docs/`:
- `Kelompok3_Manuscript.pdf`
- `Kelompok3_PPT.pdf`

## Team

Kelompok 3 — *(add names)*

## Built with

Python, PyTorch, torchaudio (WAV2VEC2_BASE), scikit-learn, XGBoost, imbalanced-learn (SMOTE), SHAP, LIME, Jupyter

## License

*(add one if you want the repo reusable, e.g. MIT)*
