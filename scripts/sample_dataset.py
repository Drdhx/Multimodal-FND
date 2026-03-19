"""
Fakeddit Dataset Sampler
Reduces train/val/test TSV files to a manageable size.

Creator : Dhanush D  |  dhanushd1812@gmail.com  |  github.com/Drdhx

Usage:
    python sample_dataset.py

Output:
    multimodal_train_small.tsv      (5000 rows, balanced)
    multimodal_validate_small.tsv   (1000 rows, balanced)
    multimodal_test_small.tsv       (1000 rows, balanced)
"""

import pandas as pd
import os

# ── Config — change these numbers to whatever size you want ──────────────────
TRAIN_SIZE = 5000   # rows for training   (balanced per class)
VAL_SIZE   = 1000   # rows for validation
TEST_SIZE  = 1000   # rows for testing

LABEL_COL  = '2_way_label'   # change to '6_way_label' if using 6-class task
SEED       = 42

DATA_DIR   = r'C:\Users\Dhanush\OneDrive\Desktop\FND\multimodal'

splits = {
    'train':    (os.path.join(DATA_DIR, 'multimodal_train.tsv'),         TRAIN_SIZE, 'multimodal_train_small.tsv'),
    'validate': (os.path.join(DATA_DIR, 'multimodal_validate.tsv'),      VAL_SIZE,   'multimodal_validate_small.tsv'),
    'test':     (os.path.join(DATA_DIR, 'multimodal_test_public.tsv'),   TEST_SIZE,  'multimodal_test_small.tsv'),
}

for split_name, (in_path, n_samples, out_name) in splits.items():
    print(f"\nProcessing {split_name}...")

    if not os.path.exists(in_path):
        print(f"  SKIP — file not found: {in_path}")
        continue

    df = pd.read_csv(in_path, sep='\t', on_bad_lines='skip')
    df = df.dropna(subset=['clean_title', LABEL_COL])
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    print(f"  Original size : {len(df):,} rows")
    print(f"  Class counts  : {df[LABEL_COL].value_counts().to_dict()}")

    # Balanced sampling — equal rows per class
    n_classes    = df[LABEL_COL].nunique()
    per_class    = n_samples // n_classes
    sampled      = (
        df.groupby(LABEL_COL, group_keys=False)
          .apply(lambda g: g.sample(min(per_class, len(g)), random_state=SEED))
          .reset_index(drop=True)
    )

    out_path = os.path.join(DATA_DIR, out_name)
    sampled.to_csv(out_path, sep='\t', index=False)

    print(f"  Sampled size  : {len(sampled):,} rows")
    print(f"  Class counts  : {sampled[LABEL_COL].value_counts().to_dict()}")
    print(f"  Saved to      : {out_path}")

print("\n\nDone! Now update CFG in your notebook:")
print(f'  TRAIN_TSV = r"{DATA_DIR}\\multimodal_train_small.tsv"')
print(f'  VAL_TSV   = r"{DATA_DIR}\\multimodal_validate_small.tsv"')
print(f'  TEST_TSV  = r"{DATA_DIR}\\multimodal_test_small.tsv"')
print(f'  MAX_SAMPLES = None   # already small, no need to cap further')
