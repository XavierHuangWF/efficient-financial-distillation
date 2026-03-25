import os
import json
import re
import random
import numpy as np
import pandas as pd
import torch
import evaluate
import matplotlib.pyplot as plt

from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split

from sentence_transformers import SentenceTransformer
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans

from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)

from LogExportCallback import LogExportCallback

print("CUDA:", torch.version.cuda)

# ===================== GLOBAL SEED =====================
SEED = 24266

# ===================== Reproducibility =====================
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ===================== EXPERIMENT SETTINGS =====================
# Data regimes:
#  1) Synthetic only
#  2) Seed only (random/kmeans/file)
#  3) Full supervised (PhraseBank)  [when both switches below are False]
# You can MERGE synthetic + seed (set both True).

NUM_EPOCHS = 50
FREEZE_LAYERS = 4
USE_EARLY_STOPPING = True

# --- data regime switches ---
USE_SYNTHETIC_DATA = False     # load synthetic JSONL and use it (alone or merged)
USE_SEED_DATA      = True      # select a seed set (random/kmeans/file) and use it (alone or merged)

# If both True  -> train on SYNTHETIC + SEED (deduped)
# If only synthetic True -> synthetic only
# If only seed True -> seed only
# If both False -> full PhraseBank supervised

# --- seed selection settings ---
USE_CLASS_BALANCED_CLUSTERING = True
SEED_SELECTION_METHOD = "file"           # "random" | "kmeans" | "file"
SAMPLES_PER_CLASS = 35
SEED_PATH = "../outputs/2seed_data_random.jsonl"  # used if SEED_SELECTION_METHOD == "file"

# --- synthetic file path ---
SYNTHETIC_PATH = "../outputs/2synthetic_data_from_randonSeed.jsonl"  # fixed typo

# --- model/tokenizer ---
MODEL_NAME = "prajjwal1/bert-tiny"
TOKENIZER_NAME = MODEL_NAME

# ===================== Cleaning =====================
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

# ===================== Plotting =====================
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

# ===================== Data Loading =====================
def DataLoad():
    dataset = load_dataset("takala/financial_phrasebank", "sentences_allagree",
                           split="train", trust_remote_code=True)
    return pd.DataFrame(dataset)

def Stratified_Split(data):
    train_df, temp_df = train_test_split(
        data, test_size=0.2, stratify=data["label"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=SEED
    )
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    for label, count in train_df["label"].value_counts().sort_index().items():
        pct = count / len(train_df) * 100
        print(f"Label {label}: {count} ({pct:.2f}%)")
    return train_df, val_df, test_df

# ===== Robust JSONL loader (vectorized; drops bad/missing labels) =====
def _load_jsonl_as_df(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame(rows)
    if not {"input", "output"}.issubset(df.columns):
        raise ValueError(f"{path} missing required keys 'input' and/or 'output'")

    df = df.rename(columns={"input": "sentence", "output": "label"})[["sentence", "label"]].copy()
    df["sentence"] = df["sentence"].astype(str).apply(clean_text)

    before = len(df)
    df = df[df["label"].notna()].copy()  # drop missing

    # vectorized normalization to {0,1,2}
    label_map = {
        "negative": 0, "bearish": 0, "neg": 0, "bear": 0, "-1": 0,
        "neutral": 1, "neu": 1, "0": 1,
        "positive": 2, "bullish": 2, "pos": 2, "bull": 2, "1": 2, "2": 2
    }
    df["label"] = (
        df["label"]
        .astype(str).str.strip().str.strip('"').str.strip("'").str.lower()
        .map(label_map)
    )
    dropped = before - df["label"].notna().sum()
    if dropped:
        print(f"[WARN] Dropped {dropped} rows in {path} with invalid/missing labels.")

    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)
    return df

# ===================== Seed Helpers =====================
def Generate_Embeddings(sentences):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model.encode(sentences, show_progress_bar=True)

def Cluster_Embeddings(embeddings, num_clusters=100):
    model = KMeans(n_clusters=num_clusters, random_state=SEED)
    model.fit(embeddings)
    return model

def _select_seeds_random(df):
    frames = []
    for label in sorted(df["label"].unique()):
        sub = df[df["label"] == label]
        n = min(SAMPLES_PER_CLASS, len(sub))
        frames.append(sub.sample(n=n, random_state=SEED, replace=False))
    seed_df = pd.concat(frames).reset_index(drop=True)
    print(f"[Random] Seed counts:\n{seed_df['label'].value_counts().sort_index()}")
    return seed_df

def _select_seeds_kmeans(df):
    frames = []
    for label in sorted(df["label"].unique()):
        subset = df[df["label"] == label].reset_index(drop=True)
        n_clusters = min(SAMPLES_PER_CLASS, len(subset))
        embeddings = Generate_Embeddings(subset["sentence"].tolist())
        cluster_model = Cluster_Embeddings(embeddings, num_clusters=n_clusters)
        closest, _ = pairwise_distances_argmin_min(cluster_model.cluster_centers_, embeddings)
        frames.append(subset.iloc[closest])
    seed_df = pd.concat(frames).reset_index(drop=True)
    print(f"[KMeans] Seed counts:\n{seed_df['label'].value_counts().sort_index()}")
    return seed_df

def Select_Seed_Data(df):
    m = SEED_SELECTION_METHOD.lower()
    if m == "file":
        print(f"[Seed] Loading from file: {SEED_PATH}")
        return _load_jsonl_as_df(SEED_PATH)
    if m == "random":
        return _select_seeds_random(df)
    # default: kmeans (class-balanced if flag set)
    if USE_CLASS_BALANCED_CLUSTERING:
        return _select_seeds_kmeans(df)
    else:
        # global 105-centroid variant (not class-balanced)
        embeddings = Generate_Embeddings(df["sentence"].tolist())
        clustering_model = Cluster_Embeddings(embeddings, num_clusters=105)
        closest, _ = pairwise_distances_argmin_min(clustering_model.cluster_centers_, embeddings)
        seed_df = df.iloc[closest].reset_index(drop=True)
        print(f"[KMeans (global 105)] Seed counts:\n{seed_df['label'].value_counts().sort_index()}")
        return seed_df

# ===================== Tokenizer / Metrics =====================
tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_NAME)

def tokenize_function(example):
    return tokenizer(example["sentence"], truncation=True)

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc  = accuracy.compute(predictions=preds, references=labels)
    prec = precision.compute(predictions=preds, references=labels, average="macro")
    rec  = recall.compute(predictions=preds, references=labels, average="macro")
    f1_s = f1.compute(predictions=preds, references=labels, average="macro")

    return {
        "accuracy": acc["accuracy"],
        "precision": prec["precision"],
        "recall": rec["recall"],
        "f1": f1_s["f1"],
    }

# ===================== MAIN =====================
if __name__ == "__main__":
    if torch.cuda.is_available():
        print("GPU device:", torch.cuda.get_device_name(0))

    # PhraseBank split (used for validation/test and possibly training)
    data = DataLoad()
    full_train_df, val_df, test_df = Stratified_Split(data)

    # ---- Build train_df per the switches ----
    train_parts = []

    if USE_SYNTHETIC_DATA:
        syn_df = _load_jsonl_as_df(SYNTHETIC_PATH)
        print(f"Synthetic training size: {len(syn_df)}")
        train_parts.append(syn_df)

    if USE_SEED_DATA:
        seed_df = Select_Seed_Data(full_train_df)
        print(f"Seed training size: {len(seed_df)}")
        train_parts.append(seed_df)

    if len(train_parts) == 0:
        # full supervised regime
        print(f"Using full training data: {len(full_train_df)} samples")
        train_df = full_train_df.reset_index(drop=True)
    else:
        # merge selected parts (synthetic and/or seed)
        train_df = pd.concat(train_parts, ignore_index=True)
        before = len(train_df)
        train_df = train_df.drop_duplicates(subset=["sentence", "label"]).reset_index(drop=True)
        print(f"Deduped {before - len(train_df)} duplicates. Final train size: {len(train_df)}")

    # Convert to HF datasets
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset   = Dataset.from_pandas(val_df.reset_index(drop=True))
    test_dataset  = Dataset.from_pandas(test_df.reset_index(drop=True))

    # Tokenize
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset   = val_dataset.map(tokenize_function, batched=True)
    test_dataset  = test_dataset.map(tokenize_function, batched=True)

    # Training args (note: evaluation_strategy is the correct arg name)
    training_args = TrainingArguments(
        seed=SEED,
        output_dir="../outputs",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-3,
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

    # Model + freezing
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, use_safetensors=True
    )
    if FREEZE_LAYERS > 0:
        print(f"Freezing first {FREEZE_LAYERS} transformer layers...")
        for param in model.bert.encoder.layer[:FREEZE_LAYERS].parameters():
            param.requires_grad = False
    else:
        print("No transformer layers are frozen. Full fine-tuning enabled.")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    log_callback = LogExportCallback()
    callbacks = [log_callback]
    if USE_EARLY_STOPPING:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=10))

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

    # Save training history
    with open("../outputs/training_history.json", "w") as f:
        json.dump(log_callback.history, f, indent=2)
    print("Training history saved to ../outputs/training_history.json")

    plot_training_metrics(log_callback)
    trainer.save_model("../outputs/best_model")

    # Evaluate on test
    from sklearn.metrics import classification_report
    pred_output = trainer.predict(test_dataset)
    preds = np.argmax(pred_output.predictions, axis=1)
    labels = pred_output.label_ids

    report = classification_report(
        labels, preds, output_dict=True,
        target_names=["Negative", "Neutral", "Positive"]
    )
    print("=== Classification Report ===")
    print(classification_report(
        labels, preds,
        target_names=["Negative", "Neutral", "Positive"],
        digits=4
    ))

    test_metrics = {
        "epoch": "test",
        "val_loss": pred_output.metrics["test_loss"],
        "train_loss": None,
        "val_accuracy": report["accuracy"],
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1": report["macro avg"]["f1-score"]
    }

    # Append to metrics CSV
    df_metrics_path = "../outputs/training_metrics.csv"
    if os.path.exists(df_metrics_path):
        df_metrics = pd.read_csv(df_metrics_path)
    else:
        df_metrics = pd.DataFrame(columns=["epoch","val_loss","train_loss","val_accuracy","precision","recall","f1"])
    df_metrics = pd.concat([df_metrics, pd.DataFrame([test_metrics])], ignore_index=True)
    df_metrics.to_csv(df_metrics_path, index=False)
    print("Test evaluation metrics appended to ../outputs/training_metrics.csv")

    with open("../outputs/test_classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Full test classification report saved to ../outputs/test_classification_report.json")
