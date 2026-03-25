import os
import pandas as pd
import torch
import numpy as np
import evaluate
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
import json
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans
from transformers import EarlyStoppingCallback

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

from LogExportCallback import LogExportCallback
import random

# =========================
# Reproducibility
# =========================
def set_seed(seed=24266):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()

# =========================
# Cleaning
# =========================
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

# =========================
# Config (match Code 1)
# =========================
MODEL_NAME = "answerdotai/ModernBERT-base"
NUM_EPOCHS = 50
FREEZE_LAYERS = 4

# Code 1 trains on SYNTHETIC + SEED (merged)
USE_SYNTHETIC_DATA = False
USE_SEED_ONLY      = True
USE_EARLY_STOPPING = True

# "file" | "random" | "kmeans"
SEED_SELECTION_METHOD = "file"
SAMPLES_PER_CLASS     = 35

SYNTHETIC_PATH = "../outputs/1synthetic_data_from_randonSeed.jsonl"
# >>> The ONLY intended difference vs Code 1:
SEED_PATH      = "../outputs/1seed_data_random.jsonl"   # random seed file

# =========================
# Tokenizer / Model (match Code 1)
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

# =========================
# Plot training metrics
# =========================
def plot_training_metrics(callback, save_path="../outputs/training_metrics.png",
                          csv_path="../outputs/training_metrics.csv"):
    history = callback.history
    epochs = history["epoch"]
    val_loss = history["val_loss"]
    train_loss = history["train_loss"][:len(epochs)]
    val_accuracy = history["val_accuracy"]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Training Metrics Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()

    df = pd.DataFrame({
        "epoch": epochs,
        "val_loss": val_loss,
        "train_loss": train_loss,
        "val_accuracy": val_accuracy
    })
    df.to_csv(csv_path, index=False)
    print(f"CSV saved to {csv_path}")

# =========================
# Data loading helpers (match Code 1)
# =========================
def DataLoad():
    dataset = load_dataset("takala/financial_phrasebank", "sentences_allagree",
                           split="train", trust_remote_code=True)
    return pd.DataFrame(dataset)

def Stratified_Split(data):
    train_df, temp_df = train_test_split(
        data, test_size=0.2, stratify=data["label"], random_state=24266
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=24266
    )
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    for label, count in train_df["label"].value_counts().items():
        percentage = count / len(train_df) * 100
        print(f"Label {label}: {count} ({percentage:.2f}%)")
    return train_df, val_df, test_df

def tokenize_function(example):
    return tokenizer(example["sentence"], truncation=True)

def compute_metrics(eval_pred):
    accuracy  = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall    = evaluate.load("recall")
    f1        = evaluate.load("f1")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc  = accuracy.compute(predictions=predictions, references=labels)
    prec = precision.compute(predictions=predictions, references=labels, average="macro")
    rec  = recall.compute(predictions=predictions, references=labels, average="macro")
    f1_s = f1.compute(predictions=predictions, references=labels, average="macro")
    return {"accuracy": acc["accuracy"],
            "precision": prec["precision"],
            "recall": rec["recall"],
            "f1": f1_s["f1"]}

# =========================
# JSONL loader (match Code 1’s title-case mapper)
# =========================
def _normalize_label_to_phrasebank_id(lbl: str) -> int:
    """
    Map to PhraseBank ids: 0=Negative, 1=Neutral, 2=Positive.
    Accept both {Negative, Neutral, Positive} and {Bearish, Neutral, Bullish}.
    """
    if not isinstance(lbl, str):
        raise ValueError(f"Label is not a string: {lbl}")
    t = lbl.strip().title()
    if t in ("Negative", "Bearish"):
        return 0
    if t == "Neutral":
        return 1
    if t in ("Positive", "Bullish"):
        return 2
    raise ValueError(f"Unexpected label value: {lbl}")

def _load_jsonl_to_df(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame(rows)
    if not {"input", "output"}.issubset(df.columns):
        raise ValueError(f"{path} missing required keys 'input' and/or 'output'")
    df = df.rename(columns={"input": "sentence", "output": "label"})[["sentence", "label"]].copy()
    df["sentence"] = df["sentence"].astype(str).apply(clean_text)
    df["label"] = df["label"].apply(_normalize_label_to_phrasebank_id).astype(int)
    return df

# =========================
# Seed selection FROM TRAIN POOL (match Code 1)
# =========================
def _select_seeds_random_from_pool(pool_df: pd.DataFrame, samples_per_class=35) -> pd.DataFrame:
    pool_df = pool_df.copy()
    pool_df["sentence"] = pool_df["sentence"].astype(str).apply(clean_text)
    chosen = []
    for cls in sorted(pool_df["label"].unique()):
        subset = pool_df[pool_df["label"] == cls].sample(
            n=min(samples_per_class, (pool_df["label"] == cls).sum()),
            random_state=24266,
            replace=False
        )
        chosen.append(subset)
    seed_df = pd.concat(chosen, ignore_index=True)
    print(f"[Random-from-train] Seed size: {len(seed_df)}")
    return seed_df[["sentence", "label"]].reset_index(drop=True)

def _select_seeds_kmeans_from_pool(pool_df: pd.DataFrame, samples_per_class=35) -> pd.DataFrame:
    pool_df = pool_df.copy()
    pool_df["sentence"] = pool_df["sentence"].astype(str).apply(clean_text)
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    selected = []
    for cls in sorted(pool_df["label"].unique()):
        subset = pool_df[pool_df["label"] == cls].reset_index(drop=True)
        n_clusters = min(samples_per_class, len(subset))
        embs = encoder.encode(subset["sentence"].tolist(), batch_size=64, show_progress_bar=False)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=24266)
        kmeans.fit(embs)
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embs)
        selected.append(subset.iloc[closest])
    seed_df = pd.concat(selected, ignore_index=True)
    print(f"[KMeans-from-train] Seed size: {len(seed_df)}")
    return seed_df[["sentence", "label"]].reset_index(drop=True)

def get_seed_df(method: str, pool_df: pd.DataFrame) -> pd.DataFrame:
    m = method.lower().strip()
    pool_df = pool_df.copy()
    pool_df["sentence"] = pool_df["sentence"].astype(str).apply(clean_text)

    if m == "file":
        print("[Seed] Using seed file:", SEED_PATH)
        file_df = _load_jsonl_to_df(SEED_PATH)  # cleaned & mapped
        merged = file_df.merge(pool_df[["sentence", "label"]], on=["sentence", "label"], how="inner")
        if len(merged) < len(file_df):
            print(f"[WARN] {len(file_df)-len(merged)} file-seed rows were not in the train pool and were dropped.")
        return merged[["sentence", "label"]].reset_index(drop=True)
    if m == "random":
        return _select_seeds_random_from_pool(pool_df, SAMPLES_PER_CLASS)
    if m == "kmeans":
        return _select_seeds_kmeans_from_pool(pool_df, SAMPLES_PER_CLASS)
    raise ValueError(f"Unknown SEED_SELECTION_METHOD: {method}")

# =========================
# Layer freezing (match Code 1 helper)
# =========================
def freeze_first_n_layers(model, n: int):
    if n <= 0:
        print("No transformer layers are frozen. Full fine-tuning enabled.")
        return
    frozen = False
    for attr_path in [
        "model.layers",             # ModernBERT common
        "modern_bert.layers",
        "base_model.layers",
        "model.encoder.layer",      # generic BERT
    ]:
        try:
            mod = model
            for part in attr_path.split("."):
                mod = getattr(mod, part)
            layers = list(mod) if isinstance(mod, (list, tuple)) else mod
            target = layers[:n] if isinstance(layers, (list, tuple)) else layers[:n]
            for p in target.parameters():
                p.requires_grad = False
            print(f"Freezing first {n} transformer layers via '{attr_path}'")
            frozen = True
            break
        except Exception:
            continue
    if not frozen:
        print("[WARN] Could not locate encoder layers to freeze; proceeding without freezing.")

# =========================
# Main (match Code 1)
# =========================
if __name__ == "__main__":
    if torch.cuda.is_available():
        print("GPU device:", torch.cuda.get_device_name(0))

    if USE_SYNTHETIC_DATA and USE_SEED_ONLY:
        raise ValueError("Set only one of USE_SYNTHETIC_DATA or USE_SEED_ONLY to True, not both.")

    # 1) Clean BEFORE split (important)
    full_data = DataLoad()
    full_data["sentence"] = full_data["sentence"].astype(str).apply(clean_text)
    train_pool_df, val_df, test_df = Stratified_Split(full_data)

    # 2) Build training set per regime
    if USE_SYNTHETIC_DATA:
        syn_df = _load_jsonl_to_df(SYNTHETIC_PATH)
        print(f"Synthetic training size (pre-merge): {len(syn_df)}")
        seed_df = get_seed_df(SEED_SELECTION_METHOD, pool_df=train_pool_df)
        print(f"Seed size: {len(seed_df)}")
        train_df = pd.concat([syn_df, seed_df], ignore_index=True)
    elif USE_SEED_ONLY:
        train_df = get_seed_df(SEED_SELECTION_METHOD, pool_df=train_pool_df)
        print(f"Seed-only training size: {len(train_df)}")
    else:
        train_df = train_pool_df.copy()

    # De-dup + leakage sanity check
    before = len(train_df)
    train_df = train_df.drop_duplicates(subset=["sentence", "label"]).reset_index(drop=True)
    if len(train_df) != before:
        print(f"Deduped {before - len(train_df)} duplicate rows (sentence+label).")

    leak_val = set(train_df["sentence"]).intersection(set(val_df["sentence"]))
    leak_test = set(train_df["sentence"]).intersection(set(test_df["sentence"]))
    if leak_val or leak_test:
        print(f"[WARN] Potential overlap: train∩val={len(leak_val)}, train∩test={len(leak_test)}")

    # 3) HF datasets
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset   = Dataset.from_pandas(val_df.reset_index(drop=True))
    test_dataset  = Dataset.from_pandas(test_df.reset_index(drop=True))

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset   = val_dataset.map(tokenize_function, batched=True)
    test_dataset  = test_dataset.map(tokenize_function, batched=True)

    # 4) Training args (match Code 1)
    training_args = TrainingArguments(
        seed=24266,
        data_seed=24266,
        output_dir="../outputs",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=32,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.1,
        label_smoothing_factor=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_dir="../outputs/logs",
        logging_strategy="epoch",
        save_total_limit=1
    )

    # 5) Freeze layers
    freeze_first_n_layers(model, FREEZE_LAYERS)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    log_callback = LogExportCallback()
    callbacks = [log_callback]
    if USE_EARLY_STOPPING:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=10))

    # 6) Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )
    trainer.train()

    # 7) Save history/plots/model
    os.makedirs("../outputs", exist_ok=True)
    with open("../outputs/training_history.json", "w") as f:
        json.dump(log_callback.history, f, indent=2)
    print("Training history saved.")

    plot_training_metrics(log_callback)
    trainer.save_model("../outputs/best_model")

    # 8) Test
    from sklearn.metrics import classification_report
    pred_output = trainer.predict(test_dataset)
    preds = np.argmax(pred_output.predictions, axis=1)
    labels = pred_output.label_ids

    report = classification_report(labels, preds, output_dict=True,
                                   target_names=["Negative", "Neutral", "Positive"])
    print("=== Classification Report ===")
    print(classification_report(labels, preds, target_names=["Negative", "Neutral", "Positive"], digits=4))

    test_metrics = {
        "epoch": "test",
        "val_loss": pred_output.metrics["test_loss"],
        "train_loss": None,
        "val_accuracy": report["accuracy"],
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1": report["macro avg"]["f1-score"]
    }

    # 9) Append to metrics CSV (match Code 1’s defensive behavior)
    metrics_csv = "../outputs/training_metrics.csv"
    if os.path.exists(metrics_csv):
        df_metrics = pd.read_csv(metrics_csv)
    else:
        df_metrics = pd.DataFrame(columns=["epoch","val_loss","train_loss","val_accuracy",
                                           "precision","recall","f1"])
    df_metrics = pd.concat([df_metrics, pd.DataFrame([test_metrics])], ignore_index=True)
    df_metrics.to_csv(metrics_csv, index=False)
    print("Test metrics appended.")

    with open("../outputs/test_classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Full test classification report saved.")
