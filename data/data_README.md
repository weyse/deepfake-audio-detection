# Dataset

This project uses the **Fake-or-Real (FoR)** dataset for training/evaluation, plus its re-recorded variant for the cross-condition robustness test.

The raw audio files are not included in this repository (too large for GitHub). To reproduce the results:

1. Download the FoR dataset. https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset - download the for-norm set (normalized, non trimmed)
2. Extract it so the folder structure looks like this, matching what the scripts expect:

```
data/
├── datasets_separated/
│   ├── real_training/  
│   ├── fake_training/
│   ├── real_validation/
│   ├── fake_validation/
│   ├── real_testing/
│   └── fake_testing/
└── for-rerec/
    └── for-rerecorded/
        └── testing/
            ├── real/
            └── fake/
```

3. Each `real_*` / `fake_*` folder should contain `outliers.csv` and `<name>_below_treshold.csv` with a `file path` and `duration` column pointing to the actual audio files — this is what `exp1_run.py` reads.

Once the data is in place, you can run `notebooks/preprocess.ipynb` followed by the scripts in `scripts/` as described in the main README.
