"""
Multimodal Fake News Detection — Evaluation Script
===================================================

Author  : Dhanush D
Email   : dhanushd1812@gmail.com
GitHub  : github.com/Drdhx

Usage:
    python evaluate.py --checkpoint results/checkpoints/best_model.pt
"""

import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, precision_recall_fscore_support
)
from transformers import DistilBertTokenizer
from tqdm import tqdm

from src.model import MultimodalFakeNewsDetector
from src.dataset import load_split, build_meta_features, build_dataloaders
from train import CFG


def evaluate(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results/plots", exist_ok=True)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    META_COLS     = ckpt["meta_cols"]
    meta_input_dim = ckpt["meta_input_dim"]

    # Load data
    train_df = load_split(CFG.TRAIN_TSV, CFG.LABEL_COL, CFG.MAX_SAMPLES)
    val_df   = load_split(CFG.VAL_TSV,   CFG.LABEL_COL)
    test_df  = load_split(CFG.TEST_TSV,  CFG.LABEL_COL)
    train_meta, val_meta, test_meta, _, scaler = \
        build_meta_features(train_df, val_df, test_df)

    tokenizer = DistilBertTokenizer.from_pretrained(CFG.TEXT_MODEL_NAME)
    _, _, test_loader = build_dataloaders(
        CFG, train_df, val_df, test_df,
        train_meta, val_meta, test_meta, tokenizer)

    # Build model and load weights
    model = MultimodalFakeNewsDetector(
        meta_input_dim=meta_input_dim,
        num_classes=CFG.NUM_CLASSES,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Run inference
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            imgs = batch["image"].to(device)
            meta = batch["metadata"].to(device)
            lbls = batch["label"].to(device)
            logits = model(ids, mask, imgs, meta)
            all_preds.append(logits.argmax(-1).cpu())
            all_labels.append(lbls.cpu())
            all_probs.append(F.softmax(logits, -1).cpu())

    y_pred  = torch.cat(all_preds).numpy()
    y_true  = torch.cat(all_labels).numpy()
    y_probs = torch.cat(all_probs).numpy()

    target_names = ["Real", "Fake"]

    # Print report
    print("=" * 55)
    print("  CLASSIFICATION REPORT — TEST SET")
    print("=" * 55)
    print(classification_report(y_true, y_pred,
          target_names=target_names, digits=4))
    auc = roc_auc_score(y_true, y_probs[:, 1])
    print(f"AUC-ROC: {auc:.4f}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Test Set Evaluation Results", fontsize=14, fontweight="bold")

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names,
                ax=axes[0, 0], annot_kws={"size": 16, "weight": "bold"})
    axes[0, 0].set_title("Confusion Matrix", fontweight="bold")
    axes[0, 0].set_xlabel("Predicted"); axes[0, 0].set_ylabel("True")

    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
    axes[0, 1].fill_between(fpr, tpr, alpha=0.15, color="steelblue")
    axes[0, 1].plot(fpr, tpr, color="steelblue", lw=2.5, label=f"AUC={auc:.4f}")
    axes[0, 1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0, 1].set_title("ROC Curve", fontweight="bold")
    axes[0, 1].set_xlabel("FPR"); axes[0, 1].set_ylabel("TPR")
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1])
    x = np.arange(2)
    axes[1, 0].bar(x - 0.25, prec, 0.25, label="Precision", color="steelblue")
    axes[1, 0].bar(x,        rec,  0.25, label="Recall",    color="mediumseagreen")
    axes[1, 0].bar(x + 0.25, f1,   0.25, label="F1",        color="tomato")
    axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(target_names)
    axes[1, 0].set_ylim(0, 1.15); axes[1, 0].set_title("Per-Class Metrics", fontweight="bold")
    axes[1, 0].legend(); axes[1, 0].grid(axis="y", alpha=0.3)

    correct = (y_pred == y_true)
    axes[1, 1].hist(y_probs[correct, 1],  bins=15, alpha=0.7,
                    label="Correct",   color="mediumseagreen", edgecolor="white")
    axes[1, 1].hist(y_probs[~correct, 1], bins=15, alpha=0.7,
                    label="Incorrect", color="tomato", edgecolor="white")
    axes[1, 1].set_title("Confidence Distribution", fontweight="bold")
    axes[1, 1].set_xlabel("P(Fake)"); axes[1, 1].set_ylabel("Count")
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/plots/results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Results plot saved to results/plots/results.png")


def predict(title: str, checkpoint_path: str, image_path: str = None):
    """Run single-sample inference."""
    from PIL import Image
    from src.dataset import VAL_TRANSFORMS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(checkpoint_path, map_location=device)

    train_df = load_split(CFG.TRAIN_TSV, CFG.LABEL_COL, CFG.MAX_SAMPLES)
    val_df   = load_split(CFG.VAL_TSV,   CFG.LABEL_COL)
    test_df  = load_split(CFG.TEST_TSV,  CFG.LABEL_COL)
    _, _, _, META_COLS, scaler = build_meta_features(train_df, val_df, test_df)

    tokenizer = DistilBertTokenizer.from_pretrained(CFG.TEXT_MODEL_NAME)
    model = MultimodalFakeNewsDetector(
        meta_input_dim=ckpt["meta_input_dim"], num_classes=CFG.NUM_CLASSES).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    enc  = tokenizer(title, max_length=CFG.MAX_LEN, padding="max_length",
                     truncation=True, return_tensors="pt")
    ids  = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)

    if image_path and os.path.exists(image_path):
        img = Image.open(image_path).convert("RGB")
    else:
        img = Image.new("RGB", (224, 224), (0, 0, 0))
    img_t  = VAL_TRANSFORMS(img).unsqueeze(0).to(device)
    meta_t = torch.zeros(1, ckpt["meta_input_dim"]).to(device)

    with torch.no_grad():
        probs = F.softmax(model(ids, mask, img_t, meta_t), -1).squeeze().cpu().numpy()

    pred = int(probs.argmax())
    label_map = {0: "Real", 1: "Fake"}
    print(f"Title     : {title}")
    print(f"Prediction: {label_map[pred]}  (confidence: {probs.max():.3f})")
    print(f"Probs     : Real={probs[0]:.3f}  Fake={probs[1]:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/checkpoints/best_model.pt")
    parser.add_argument("--predict", type=str, default=None,
                        help="Run single prediction on a headline string")
    args = parser.parse_args()

    if args.predict:
        predict(args.predict, args.checkpoint)
    else:
        evaluate(args.checkpoint)
