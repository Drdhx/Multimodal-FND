"""
Multimodal Fake News Detection — Dataset & DataLoader
======================================================

Author  : Dhanush D
Email   : dhanushd1812@gmail.com
GitHub  : github.com/Drdhx
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from transformers import DistilBertTokenizer
from sklearn.preprocessing import StandardScaler
from typing import List, Optional


# ── Image transforms ──────────────────────────────────────────────────────────

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Metadata feature columns ──────────────────────────────────────────────────

META_NUMERIC = ["score", "num_comments", "upvote_ratio"]
META_BINARY  = ["is_video", "over_18", "has_image"]


# ── Dataset ───────────────────────────────────────────────────────────────────

class FakedditDataset(Dataset):
    """
    Multimodal Fakeddit dataset.

    Returns per sample:
        input_ids      : [max_len]       text token ids
        attention_mask : [max_len]       text attention mask
        image          : [3, 224, 224]   image tensor
        metadata       : [meta_dim]      scaled metadata features
        label          : scalar          class label
    """

    def __init__(
        self,
        df: pd.DataFrame,
        meta_array: np.ndarray,
        tokenizer: DistilBertTokenizer,
        image_dir: str,
        transform,
        label_col: str = "2_way_label",
        max_len: int = 128,
    ):
        self.df        = df.reset_index(drop=True)
        self.meta      = meta_array.astype(np.float32)
        self.tokenizer = tokenizer
        self.image_dir = image_dir
        self.transform = transform
        self.label_col = label_col
        self.max_len   = max_len
        self._blank    = Image.new("RGB", (224, 224), (0, 0, 0))

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, post_id: str) -> Image.Image:
        path = os.path.join(self.image_dir, f"{post_id}.jpg")
        try:
            return Image.open(path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, Exception):
            return self._blank

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            str(row["clean_title"]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        img = self._load_image(str(row.get("id", "")))
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "image":          self.transform(img),
            "metadata":       torch.tensor(self.meta[idx], dtype=torch.float32),
            "label":          torch.tensor(int(row[self.label_col]), dtype=torch.long),
        }


# ── Data loading helpers ──────────────────────────────────────────────────────

def load_split(path: str, label_col: str,
               max_samples: Optional[int] = None,
               seed: int = 42) -> pd.DataFrame:
    """Load and clean a TSV split file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"TSV not found: {path}")
    df = pd.read_csv(path, sep="\t", on_bad_lines="skip", low_memory=False)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    if "clean_title" not in df.columns or label_col not in df.columns:
        raise KeyError(f"Missing 'clean_title' or '{label_col}' in {path}")
    df = df.dropna(subset=["clean_title", label_col])
    df[label_col] = df[label_col].astype(int)
    df["title_len"] = df["clean_title"].str.split().str.len().fillna(0)
    if max_samples:
        df = df.sample(min(max_samples, len(df)),
                       random_state=seed).reset_index(drop=True)
    return df


def build_meta_features(train_df: pd.DataFrame,
                        val_df: pd.DataFrame,
                        test_df: pd.DataFrame):
    """
    Engineer and scale metadata features.
    Returns (train_meta, val_meta, test_meta, META_COLS, scaler)
    """
    # Subreddit frequency encoding (log1p to reduce dominant subreddit impact)
    if "subreddit" in train_df.columns:
        freq = train_df["subreddit"].fillna("unknown").value_counts().to_dict()
        for df in [train_df, val_df, test_df]:
            df["subreddit_enc"] = np.log1p(
                df["subreddit"].fillna("unknown").map(freq).fillna(0))

    # Image availability flag
    for df in [train_df, val_df, test_df]:
        if "has_image" not in df.columns and "hasImage" in df.columns:
            df["has_image"] = (df["hasImage"].astype(str)
                               .str.lower().eq("true").astype(float))

    META_COLS = []
    candidates = META_NUMERIC + META_BINARY + ["title_len", "subreddit_enc"]
    for c in candidates:
        if c in train_df.columns:
            META_COLS.append(c)

    for df in [train_df, val_df, test_df]:
        for c in META_COLS:
            if c not in df.columns:
                df[c] = 0.0
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    scaler     = StandardScaler()
    train_meta = scaler.fit_transform(train_df[META_COLS].astype(float))
    val_meta   = scaler.transform(val_df[META_COLS].astype(float))
    test_meta  = scaler.transform(test_df[META_COLS].astype(float))
    return train_meta, val_meta, test_meta, META_COLS, scaler


def build_dataloaders(cfg, train_df, val_df, test_df,
                      train_meta, val_meta, test_meta,
                      tokenizer) -> tuple:
    """Build and return (train_loader, val_loader, test_loader)."""
    train_ds = FakedditDataset(
        train_df, train_meta, tokenizer, cfg.IMAGE_DIR,
        TRAIN_TRANSFORMS, cfg.LABEL_COL, cfg.MAX_LEN)
    val_ds   = FakedditDataset(
        val_df,   val_meta,   tokenizer, cfg.IMAGE_DIR,
        VAL_TRANSFORMS,   cfg.LABEL_COL, cfg.MAX_LEN)
    test_ds  = FakedditDataset(
        test_df,  test_meta,  tokenizer, cfg.IMAGE_DIR,
        VAL_TRANSFORMS,   cfg.LABEL_COL, cfg.MAX_LEN)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=True,  num_workers=cfg.NUM_WORKERS,
                              pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=cfg.NUM_WORKERS,
                              pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=cfg.NUM_WORKERS,
                              pin_memory=pin)
    return train_loader, val_loader, test_loader
