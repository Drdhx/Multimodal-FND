"""
Multimodal Fake News Detection — Training Script
=================================================

Author  : Dhanush D
Email   : dhanushd1812@gmail.com
GitHub  : github.com/Drdhx

Usage:
    python train.py

Update the CFG class paths before running.
"""

import os
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import DistilBertTokenizer, get_linear_schedule_with_warmup
from torchmetrics import Accuracy, F1Score
from tqdm import tqdm

from src.model import MultimodalFakeNewsDetector
from src.dataset import load_split, build_meta_features, build_dataloaders

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG  —  update paths before running
# ═══════════════════════════════════════════════════════════════════════════════

class CFG:
    # Paths
    DATA_DIR        = "data/sample"
    IMAGE_DIR       = "data/images"
    TRAIN_TSV       = os.path.join(DATA_DIR, "multimodal_train_small.tsv")
    VAL_TSV         = os.path.join(DATA_DIR, "multimodal_validate_small.tsv")
    TEST_TSV        = os.path.join(DATA_DIR, "multimodal_test_small.tsv")
    SAVE_DIR        = "results/checkpoints"

    # Task
    NUM_CLASSES     = 2
    LABEL_COL       = "2_way_label"

    # Text
    TEXT_MODEL_NAME = "distilbert-base-uncased"
    MAX_LEN         = 128
    TEXT_DROPOUT    = 0.3

    # Image
    IMG_MODEL_NAME  = "tf_efficientnetv2_s"
    IMG_SIZE        = 224
    IMG_DROPOUT     = 0.3

    # Metadata
    META_HIDDEN     = [128, 64]
    META_DIM        = 64
    META_DROPOUT    = 0.3

    # Fusion
    FUSION_DIM      = 256

    # Training
    BATCH_SIZE      = 16
    EPOCHS          = 10
    LR_BERT         = 2e-5
    LR_IMG          = 1e-4
    LR_HEAD         = 3e-4
    WEIGHT_DECAY    = 1e-2
    WARMUP_RATIO    = 0.1
    MAX_SAMPLES     = None
    NUM_WORKERS     = 0
    SEED            = 42


# ── Reproducibility ───────────────────────────────────────────────────────────

def seed_everything(seed: int = CFG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ── Training epoch ────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, scheduler,
              acc_metric, f1_metric, device, train: bool = True):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, leave=False, desc="Train" if train else "Val  ")
        for batch in pbar:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            imgs  = batch["image"].to(device)
            meta  = batch["metadata"].to(device)
            lbls  = batch["label"].to(device)

            logits = model(ids, mask, imgs, meta)
            loss   = criterion(logits, lbls)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item() * len(lbls)
            all_preds.append(logits.argmax(-1))
            all_labels.append(lbls)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    preds  = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    avg_loss = total_loss / len(loader.dataset)
    acc = acc_metric(preds, labels).item()
    f1  = f1_metric(preds, labels).item()
    acc_metric.reset(); f1_metric.reset()
    return avg_loss, acc, f1


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    seed_everything()
    os.makedirs(CFG.SAVE_DIR, exist_ok=True)
    os.makedirs(CFG.IMAGE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("\nLoading data...")
    train_df = load_split(CFG.TRAIN_TSV, CFG.LABEL_COL, CFG.MAX_SAMPLES)
    val_df   = load_split(CFG.VAL_TSV,   CFG.LABEL_COL)
    test_df  = load_split(CFG.TEST_TSV,  CFG.LABEL_COL)
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # Metadata features
    train_meta, val_meta, test_meta, META_COLS, scaler = \
        build_meta_features(train_df, val_df, test_df)
    meta_input_dim = train_meta.shape[1]
    print(f"  Metadata dim: {meta_input_dim} — {META_COLS}")

    # Tokenizer and dataloaders
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.TEXT_MODEL_NAME)
    train_loader, val_loader, test_loader = build_dataloaders(
        CFG, train_df, val_df, test_df,
        train_meta, val_meta, test_meta, tokenizer)
    print(f"  Batches — Train: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")

    # Model
    print("\nBuilding model...")
    model = MultimodalFakeNewsDetector(
        meta_input_dim=meta_input_dim,
        num_classes=CFG.NUM_CLASSES,
        fusion_dim=CFG.FUSION_DIM,
        text_model=CFG.TEXT_MODEL_NAME,
        img_model=CFG.IMG_MODEL_NAME,
        meta_hidden=CFG.META_HIDDEN,
        meta_dim=CFG.META_DIM,
        text_dropout=CFG.TEXT_DROPOUT,
        img_dropout=CFG.IMG_DROPOUT,
        meta_dropout=CFG.META_DROPOUT,
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total:,}")

    # Loss with class weights
    from collections import Counter
    counts = np.array([v for _, v in sorted(Counter(train_df[CFG.LABEL_COL]).items())])
    weights = torch.tensor(1.0 / (counts / counts.sum()), dtype=torch.float32).to(device)
    weights /= weights.sum()
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Optimizer — differential learning rates
    optimizer = AdamW([
        {"params": model.text_encoder.bert.parameters(),    "lr": CFG.LR_BERT},
        {"params": model.text_encoder.proj.parameters(),    "lr": CFG.LR_HEAD},
        {"params": model.img_encoder.backbone.parameters(), "lr": CFG.LR_IMG},
        {"params": model.img_encoder.proj.parameters(),     "lr": CFG.LR_HEAD},
        {"params": model.meta_encoder.parameters(),         "lr": CFG.LR_HEAD},
        {"params": model.fusion.parameters(),               "lr": CFG.LR_HEAD},
        {"params": model.classifier.parameters(),           "lr": CFG.LR_HEAD},
    ], weight_decay=CFG.WEIGHT_DECAY)

    total_steps  = len(train_loader) * CFG.EPOCHS
    warmup_steps = int(total_steps * CFG.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    acc_metric = Accuracy(task="multiclass", num_classes=CFG.NUM_CLASSES).to(device)
    f1_metric  = F1Score(task="multiclass", num_classes=CFG.NUM_CLASSES,
                         average="macro").to(device)

    # Training loop
    best_val_f1 = 0.0
    best_ckpt   = os.path.join(CFG.SAVE_DIR, "best_model.pt")
    history     = {"train_loss": [], "val_loss": [],
                   "train_acc":  [], "val_acc":  [],
                   "train_f1":   [], "val_f1":   []}

    print(f"\nTraining for {CFG.EPOCHS} epochs on {device}...\n")
    for epoch in range(1, CFG.EPOCHS + 1):
        t_loss, t_acc, t_f1 = run_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            acc_metric, f1_metric, device, train=True)
        v_loss, v_acc, v_f1 = run_epoch(
            model, val_loader, criterion, optimizer, scheduler,
            acc_metric, f1_metric, device, train=False)

        for k, v in [("train_loss", t_loss), ("val_loss", v_loss),
                     ("train_acc",  t_acc),  ("val_acc",  v_acc),
                     ("train_f1",   t_f1),   ("val_f1",   v_f1)]:
            history[k].append(v)

        flag = ""
        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            torch.save(model.state_dict(), best_ckpt)
            flag = "  << best"

        print(f"Epoch {epoch:02d}/{CFG.EPOCHS} | "
              f"Train Loss:{t_loss:.4f} Acc:{t_acc:.4f} F1:{t_f1:.4f} | "
              f"Val Loss:{v_loss:.4f} Acc:{v_acc:.4f} F1:{v_f1:.4f}{flag}")

    # Save full checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "meta_cols":        META_COLS,
        "meta_input_dim":   meta_input_dim,
        "best_val_f1":      best_val_f1,
        "history":          history,
        "cfg":              CFG.__dict__,
    }, os.path.join(CFG.SAVE_DIR, "full_checkpoint.pt"))
    tokenizer.save_pretrained(os.path.join(CFG.SAVE_DIR, "tokenizer"))

    print(f"\nTraining complete. Best Val F1: {best_val_f1:.4f}")
    print(f"Checkpoint saved to: {CFG.SAVE_DIR}")


if __name__ == "__main__":
    main()
