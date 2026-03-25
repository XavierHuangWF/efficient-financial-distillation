# eval_and_log_errors.py
# ------------------------------------------------------------
# Evaluate a saved model on the PhraseBank test set, log errors,
# and save confusion matrices (counts + row-normalized).
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
from sklearn.metrics import confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding,
)

# ----------------------------
# Config (edit if needed)
# ----------------------------
SEED = 24266
MODEL_DIR = "../outputs/best_model"   # where trainer.save_model() wrote the model
OUT_DIR   = "../outputs"              # where reports/plots will be saved
LABEL_NAMES = ["Negative", "Neutral", "Positive"]  # id mapping: 0,1,2
HF_DATASET = ("takala/financial_phrasebank", "sentences_allagree")  # PhraseBank split

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

def load_phrasebank_df() -> pd.DataFrame:
    ds = load_dataset(HF_DATASET[0], HF_DATASET[1], split="train", trust_remote_code=True)
    df = pd.DataFrame(ds)
    if "sentence" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset missing 'sentence' and/or 'label'.")
    df["sentence"] = df["sentence"].apply(clean_text)
    return df

def stratified_split(df: pd.DataFrame, seed: int = SEED):
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=seed)
    val_df, test_df   = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

def tokenize_fn_factory(tokenizer):
    def _fn(example):
        return tokenizer(example["sentence"], truncation=True)
    return _fn

def plot_and_save_confusion_matrices(y_true, y_pred, labels, out_dir, fname_prefix="confusion_matrix"):
    """
    Saves:
      - {fname_prefix}_counts.csv / .png
      - {fname_prefix}_row_normalized.csv / .png
    Uses fixed color scales for intuitive shading (counts: [0,max], normalized: [0,1]).
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

        # grid
        ax.set_xticks(np.arange(-.5, len(labels), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)

        # annotations (contrast-aware)
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

    _plot(cm, "Confusion Matrix (counts)", f"{fname_prefix}_counts.png",
          annotate_percent=False, vmin=0, vmax=max(1, cm.max()), cmap="Blues")

    _plot(cm_norm, "Confusion Matrix (row-normalized)", f"{fname_prefix}_row_normalized.png",
          annotate_percent=True, vmin=0.0, vmax=1.0, cmap="Blues")

# ----------------------------
# Main
# ----------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load tokenizer from saved model (fallback to base if needed)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    except Exception:
        fallback = "answerdotai/ModernBERT-base"
        try:
            cfg = json.load(open(os.path.join(MODEL_DIR, "config.json"), "r"))
            fallback = cfg.get("_name_or_path", fallback)
        except Exception:
            pass
        tokenizer = AutoTokenizer.from_pretrained(fallback)

    print(f"Loading model from {MODEL_DIR} ...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    # Rebuild the same test split (same seed + stratification)
    full_df = load_phrasebank_df()
    _, _, test_df = stratified_split(full_df, SEED)

    # Tokenize
    test_hf = Dataset.from_pandas(test_df.reset_index(drop=True))
    test_hf = test_hf.map(tokenize_fn_factory(tokenizer), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Predict
    trainer = Trainer(model=model, tokenizer=tokenizer, data_collator=data_collator)
    print("Running prediction on test set ...")
    pred_output = trainer.predict(test_hf)
    logits = pred_output.predictions
    y_true = np.array(test_hf["label"])
    y_pred = np.argmax(logits, axis=1)
    probs  = F.softmax(torch.tensor(logits), dim=-1).numpy()

    # Confusion matrices
    plot_and_save_confusion_matrices(
        y_true=y_true,
        y_pred=y_pred,
        labels=LABEL_NAMES,
        out_dir=OUT_DIR,
        fname_prefix="confusion_matrix_test"
    )

    # Misclassified samples
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
                "prob_negative": float(probs[i, 0]),
                "prob_neutral": float(probs[i, 1]),
                "prob_positive": float(probs[i, 2]),
                "pred_confidence": float(np.max(probs[i])),
                "logits": logits[i].tolist(),
            })

    mis_csv = os.path.join(OUT_DIR, "misclassified_samples.csv")
    mis_jsonl = os.path.join(OUT_DIR, "misclassified_samples.jsonl")
    if mis_records:
        pd.DataFrame(mis_records).to_csv(mis_csv, index=False)
        with open(mis_jsonl, "w", encoding="utf-8") as f:
            for r in mis_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved {len(mis_records)} misclassified samples:\n  - {mis_csv}\n  - {mis_jsonl}")
    else:
        print("No misclassifications found (perfect accuracy).")

    # Correct predictions (optional)
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
                "prob_negative": float(probs[i, 0]),
                "prob_neutral": float(probs[i, 1]),
                "prob_positive": float(probs[i, 2]),
                "pred_confidence": float(np.max(probs[i])),
            })
    corr_csv = os.path.join(OUT_DIR, "correct_samples.csv")
    pd.DataFrame(corr_records).to_csv(corr_csv, index=False)
    print(f"Saved {len(corr_records)} correct predictions to: {corr_csv}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("GPU device:", torch.cuda.get_device_name(0))
    main()
