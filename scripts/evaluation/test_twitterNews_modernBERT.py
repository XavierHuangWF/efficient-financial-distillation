# eval_twitter_only.py
# ------------------------------------------------------------
# Load a saved model and evaluate on the Twitter Financial News
# test split (no training). Saves confusion matrices and
# misclassified samples.
# ------------------------------------------------------------

import os
import re
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding,
)

# ----------------------------
# Config
# ----------------------------
SEED = 24266
MODEL_DIR = "../outputs/best_model"   # <- change if needed
OUT_DIR   = "../outputs"

# Zeroshot Twitter Financial News label mapping
# HF ids: 0=Bearish, 1=Bullish, 2=Neutral
LABEL_NAME_BY_ID = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
LABEL_ID_BY_NAME = {"Bearish": 0, "Bullish": 1, "Neutral": 2}
LABEL_NAMES      = [LABEL_NAME_BY_ID[i] for i in range(3)]

# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# ----------------------------
# Helpers
# ----------------------------
def clean_text(text):
    if not isinstance(text, str):
        return text
    text = text.replace('\\\\', '\\')
    try:
        text = bytes(text, "utf-8").decode("unicode_escape")
    except Exception:
        pass
    text = re.sub(r"^\d+\.\s*", "", text)
    text = text.replace('"', "")
    return text.strip()

TEXT_CANDIDATES = ["sentence", "text", "tweet", "content", "document", "message", "headline"]

def _find_text_column(df: pd.DataFrame) -> str:
    for c in TEXT_CANDIDATES:
        if c in df.columns:
            return c
    for c in df.columns:
        if df[c].dtype == object:
            return c
    raise ValueError("Could not find a text column in dataset.")

def load_twitter_df() -> pd.DataFrame:
    """Load zeroshot/twitter-financial-news-sentiment train+validation and clean."""
    ds_train = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train", trust_remote_code=True)
    ds_val   = load_dataset("zeroshot/twitter-financial-news-sentiment", split="validation", trust_remote_code=True)
    df = pd.concat([pd.DataFrame(ds_train), pd.DataFrame(ds_val)], ignore_index=True)

    if "label_text" in df.columns:
        lt = df["label_text"].str.strip().str.title()
        if not set(lt.unique()).issubset(set(LABEL_ID_BY_NAME.keys())):
            raise ValueError(f"Unexpected label_text values: {lt.unique()}")
        df["label"] = lt.map(LABEL_ID_BY_NAME).astype(int)
    else:
        if "label" not in df.columns:
            raise ValueError("Dataset missing 'label' column.")
        if not set(df["label"].unique()).issubset({0, 1, 2}):
            raise ValueError(f"Unexpected numeric labels: {df['label'].unique()}")
        df["label"] = df["label"].astype(int)

    text_col = _find_text_column(df)
    if text_col != "sentence":
        df = df.rename(columns={text_col: "sentence"})
    df = df[["sentence", "label"]].copy()
    df["sentence"] = df["sentence"].astype(str).apply(clean_text)
    return df

def stratified_split(df: pd.DataFrame, seed: int = SEED):
    """Recreate the same split logic used during training."""
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=seed)
    val_df, test_df   = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

def tokenize_fn_factory(tokenizer):
    def _fn(example):
        return tokenizer(example["sentence"], truncation=True)
    return _fn

def plot_and_save_confusion_matrices(y_true, y_pred, labels, out_dir, fname_prefix="twitter_confusion_matrix"):
    """
    Saves:
      - {fname_prefix}_counts.csv / .png
      - {fname_prefix}_row_normalized.csv / .png
    Fixed color scales (counts: [0,max]; normalized: [0,1]) for intuitive shading.
    """
    os.makedirs(out_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

    idx = [f"true_{l}" for l in labels]
    cols = [f"pred_{l}" for l in labels]
    pd.DataFrame(cm, index=idx, columns=cols).to_csv(os.path.join(out_dir, f"{fname_prefix}_counts.csv"))
    pd.DataFrame(cm_norm, index=idx, columns=cols).to_csv(os.path.join(out_dir, f"{fname_prefix}_row_normalized.csv"))

    def _plot(cm_plot, title, png_name, annotate_percent=False, vmin=0.0, vmax=1.0, cmap="Blues"):
        fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
        im = ax.imshow(cm_plot, cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(title)

        ax.set_xticks(np.arange(-.5, len(labels), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)

        # contrast-aware annotations
        for i in range(cm_plot.shape[0]):
            for j in range(cm_plot.shape[1]):
                if annotate_percent:
                    text = f"{cm_plot[i, j]*100:.1f}%\n({cm[i, j]})"
                    val_for_contrast = cm_plot[i, j]
                else:
                    text = f"{cm[i, j]}"
                    val_for_contrast = (cm_plot[i, j] - vmin) / (vmax - vmin) if vmax > vmin else 0
                color = "white" if val_for_contrast >= 0.5 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, png_name))
        plt.close(fig)

    # Counts: [0, max]
    _plot(cm, "Confusion Matrix (counts)", f"{fname_prefix}_counts.png",
          annotate_percent=False, vmin=0, vmax=max(1, cm.max()), cmap="Blues")

    # Row-normalized: [0, 1]
    _plot(cm_norm, "Confusion Matrix (row-normalized)", f"{fname_prefix}_row_normalized.png",
          annotate_percent=True, vmin=0.0, vmax=1.0, cmap="Blues")

# ----------------------------
# Main
# ----------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load tokenizer/model from saved directory
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    except Exception:
        # fallback: try to infer base model from config
        cfg_path = os.path.join(MODEL_DIR, "config.json")
        base_name = "answerdotai/ModernBERT-base"
        if os.path.exists(cfg_path):
            try:
                cfg = json.load(open(cfg_path, "r"))
                base_name = cfg.get("_name_or_path", base_name)
            except Exception:
                pass
        tokenizer = AutoTokenizer.from_pretrained(base_name)

    print(f"Loading model from: {MODEL_DIR}")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    # Build the same test split
    full_df = load_twitter_df()
    _, _, test_df = stratified_split(full_df, seed=SEED)

    # HF dataset + tokenize
    test_hf = Dataset.from_pandas(test_df.reset_index(drop=True))
    test_hf = test_hf.map(tokenize_fn_factory(tokenizer), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Predict
    trainer = Trainer(model=model, tokenizer=tokenizer, data_collator=data_collator)
    print("Predicting on Twitter test set ...")
    pred_output = trainer.predict(test_hf)
    logits = pred_output.predictions
    y_true = np.array(test_hf["label"])
    y_pred = np.argmax(logits, axis=1)
    probs  = F.softmax(torch.tensor(logits), dim=-1).numpy()

    # Report
    rep_dict = classification_report(y_true, y_pred, target_names=LABEL_NAMES, output_dict=True, digits=4)
    print("=== Classification Report (Twitter test) ===")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=4))
    with open(os.path.join(OUT_DIR, "twitter_test_classification_report.json"), "w") as f:
        json.dump(rep_dict, f, indent=2)

    # Confusion matrices
    plot_and_save_confusion_matrices(
        y_true=y_true,
        y_pred=y_pred,
        labels=LABEL_NAMES,
        out_dir=OUT_DIR,
        fname_prefix="twitter_confusion_matrix_test"
    )

    # Misclassified samples (+ probabilities)
    mis_records = []
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        if int(t) != int(p):
            row = test_df.iloc[i]
            mis_records.append({
                "idx": int(i),
                "sentence": row["sentence"],
                "true_id": int(t),
                "true_label": LABEL_NAMES[int(t)],
                "pred_id": int(p),
                "pred_label": LABEL_NAMES[int(p)],
                "prob_bearish":  float(probs[i, 0]),
                "prob_bullish":  float(probs[i, 1]),
                "prob_neutral":  float(probs[i, 2]),
                "pred_confidence": float(np.max(probs[i])),
                "logits": logits[i].tolist(),
            })

    mis_csv = os.path.join(OUT_DIR, "twitter_misclassified_samples.csv")
    mis_jsonl = os.path.join(OUT_DIR, "twitter_misclassified_samples.jsonl")
    if mis_records:
        pd.DataFrame(mis_records).to_csv(mis_csv, index=False)
        with open(mis_jsonl, "w", encoding="utf-8") as f:
            for r in mis_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved {len(mis_records)} misclassified samples:\n  - {mis_csv}\n  - {mis_jsonl}")
    else:
        print("No misclassifications found.")

    # (Optional) Correct predictions
    corr_records = []
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        if int(t) == int(p):
            row = test_df.iloc[i]
            corr_records.append({
                "idx": int(i),
                "sentence": row["sentence"],
                "true_id": int(t),
                "true_label": LABEL_NAMES[int(t)],
                "pred_id": int(p),
                "pred_label": LABEL_NAMES[int(p)],
                "prob_bearish":  float(probs[i, 0]),
                "prob_bullish":  float(probs[i, 1]),
                "prob_neutral":  float(probs[i, 2]),
                "pred_confidence": float(np.max(probs[i])),
            })
    corr_csv = os.path.join(OUT_DIR, "twitter_correct_samples.csv")
    pd.DataFrame(corr_records).to_csv(corr_csv, index=False)
    print(f"Saved {len(corr_records)} correct predictions to: {corr_csv}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("GPU device:", torch.cuda.get_device_name(0))
    main()
