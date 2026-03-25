import os
import json
import re
import random
import itertools
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import EarlyStoppingCallback
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import evaluate

# If this is a local helper you already have
from LogExportCallback import LogExportCallback

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
# Config
# =========================
MODEL_NAME = "answerdotai/ModernBERT-base"

NUM_EPOCHS = 50
FREEZE_LAYERS = 4
USE_SYNTHETIC_DATA =  True  # train on synthetic + seed data (merged)
USE_SEED_ONLY     =  False  # train on seed data only
USE_EARLY_STOPPING = True

SYNTHETIC_PATH = "../outputs/synthetic_data_from_Seed.jsonl"
SEED_PATH      = "../outputs/seed_data.jsonl"

TEXT_CANDIDATES = ["sentence", "text", "tweet", "content", "document", "message", "headline"]

LABEL_NAME_BY_ID = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
LABEL_ID_BY_NAME = {"Bearish": 0, "Bullish": 1, "Neutral": 2}

# =========================
# Utilities
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

def _find_text_column(df: pd.DataFrame) -> str:
    for c in TEXT_CANDIDATES:
        if c in df.columns:
            return c
    for c in df.columns:
        if df[c].dtype == object:
            return c
    raise ValueError("Could not find a text column in dataset.")

def _normalize_label_to_id(x) -> int:
    """
    Maps Bearish/Bullish/Neutral or Negative/Positive/Neutral to 0/1/2.
    Accepts strings; drops/raises on missing.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        raise ValueError("Label is NaN/None.")
    if not isinstance(x, str):
        x = str(x)

    y = x.strip().title()
    if y == "Negative": y = "Bearish"
    if y == "Positive": y = "Bullish"
    if y not in LABEL_ID_BY_NAME:
        raise ValueError(f"Unexpected label value: {x!r}")
    return LABEL_ID_BY_NAME[y]

def _load_jsonl_to_train_df(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame(rows)

    if not {"input", "output"}.issubset(df.columns):
        raise ValueError(f"{path} missing required keys 'input' and/or 'output'")

    df = df.rename(columns={"input": "sentence", "output": "label"})[["sentence", "label"]].copy()
    df["sentence"] = df["sentence"].astype(str).apply(clean_text)

    # log and drop missing/blank labels instead of crashing
    good_mask = df["label"].notna() & (df["label"].astype(str).str.strip() != "")
    bad_df = df.loc[~good_mask]
    if len(bad_df):
        os.makedirs("../outputs", exist_ok=True)
        bad_path = "../outputs/_dropped_rows_missing_label.csv"
        bad_df.to_csv(bad_path, index=False)
        print(f"[WARN] Dropping {len(bad_df)} rows with missing/empty labels. Saved preview to {bad_path}")

    df = df.loc[good_mask].copy()
    df["label"] = df["label"].apply(_normalize_label_to_id).astype(int)

    # Dedup exact pairs
    before = len(df)
    df = df.drop_duplicates(subset=["sentence", "label"]).reset_index(drop=True)
    deduped = before - len(df)
    if deduped:
        print(f"[INFO] Deduped {deduped} (sentence,label) pairs in {path}")

    uniques = sorted(df["label"].unique().tolist())
    if set(uniques) - {0, 1, 2}:
        raise ValueError(f"Unexpected numeric labels after normalize: {uniques}")

    return df

# =========================
# HF dataset load (zeroshot): concat train+validation; later do your own split
# =========================
def DataLoad():
    ds_train = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train", trust_remote_code=True)
    ds_val   = load_dataset("zeroshot/twitter-financial-news-sentiment", split="validation", trust_remote_code=True)
    df = pd.concat([pd.DataFrame(ds_train), pd.DataFrame(ds_val)], ignore_index=True)

    if "label_text" in df.columns:
        lt = df["label_text"].str.strip().str.title()  # 'Bearish','Bullish','Neutral'
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
        print(f"Label {label} ({LABEL_NAME_BY_ID[label]}): {count} ({percentage:.2f}%)")
    return train_df, val_df, test_df

# =========================
# Tokenizer (create once)
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

def tokenize_function(batch, tok=tokenizer):
    # 'batch' is a dict of lists; return tokenized dict
    return tok(batch["sentence"], truncation=True)

# =========================
# Metrics
# =========================
def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy.compute(predictions=predictions, references=labels)
    prec = precision.compute(predictions=predictions, references=labels, average="macro")
    rec = recall.compute(predictions=predictions, references=labels, average="macro")
    f1_score = f1.compute(predictions=predictions, references=labels, average="macro")

    return {
        "accuracy": acc["accuracy"],
        "precision": prec["precision"],
        "recall": rec["recall"],
        "f1": f1_score["f1"]
    }

# =========================
# Adaptive layer freezing
# =========================
def freeze_n_transformer_layers(model, n):
    if n <= 0:
        print("No transformer layers are frozen. Full fine-tuning enabled.")
        return

    candidates = [
        ("model.layers", lambda m: getattr(getattr(m, "model", m), "layers", None)),
        ("model.encoder.layers", lambda m: getattr(getattr(getattr(m, "model", m), "encoder", m), "layers", None)),
        ("encoder.layer", lambda m: getattr(getattr(m, "encoder", m), "layer", None)),  # BERT-style
    ]
    for name, getter in candidates:
        layers = getter(model)
        if layers is not None:
            L = len(layers)
            k = min(n, L)
            for p in itertools.chain.from_iterable(l.parameters() for l in layers[:k]):
                p.requires_grad = False
            print(f"Freezing first {k}/{L} transformer layers via path '{name}'.")
            return
    print("[WARN] Could not locate transformer layers to freeze; proceeding without freezing.")

# =========================
# Main
# =========================
if __name__ == "__main__":
    if torch.cuda.is_available():
        print("GPU device:", torch.cuda.get_device_name(0))

    # -------- Data selection --------
    if USE_SYNTHETIC_DATA:
        syn_df = _load_jsonl_to_train_df(SYNTHETIC_PATH)
        print(f"Synthetic rows (after cleaning): {len(syn_df)}")

        if os.path.exists(SEED_PATH):
            seed_df = _load_jsonl_to_train_df(SEED_PATH)
            print(f"Seed rows (after cleaning): {len(seed_df)}")
            train_df = pd.concat([syn_df, seed_df], ignore_index=True)

            # Drop exact duplicate (sentence, label) pairs
            before = len(train_df)
            train_df = train_df.drop_duplicates(subset=["sentence", "label"]).reset_index(drop=True)
            after = len(train_df)
            if after != before:
                print(f"Deduped {before - after} duplicate rows (sentence+label) after merge.")
        else:
            print(f"[WARN] seed file not found at {SEED_PATH}; training on synthetic only.")
            train_df = syn_df

        full_data = DataLoad()
        _, val_df, test_df = Stratified_Split(full_data)

    elif USE_SEED_ONLY:
        train_df = _load_jsonl_to_train_df(SEED_PATH)
        print(f"Seed-only training size: {len(train_df)}")
        full_data = DataLoad()
        _, val_df, test_df = Stratified_Split(full_data)

    else:
        full_data = DataLoad()
        train_df, val_df, test_df = Stratified_Split(full_data)
        for df_ in (train_df, val_df, test_df):
            df_["sentence"] = df_["sentence"].apply(clean_text)

    # -------- Datasets --------
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset   = Dataset.from_pandas(val_df.reset_index(drop=True))
    test_dataset  = Dataset.from_pandas(test_df.reset_index(drop=True))

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset   = val_dataset.map(tokenize_function, batched=True)
    test_dataset  = test_dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # -------- Model --------
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        trust_remote_code=True
    )

    if FREEZE_LAYERS > 0:
        freeze_n_transformer_layers(model, FREEZE_LAYERS)
    else:
        print("No transformer layers are frozen. Full fine-tuning enabled.")

    # -------- Training --------
    training_args = TrainingArguments(
        seed=24266,
        data_seed=24266,
        output_dir="../outputs",
        eval_strategy="epoch",   # fixed
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

    log_callback = LogExportCallback()
    trainer_callbacks = [log_callback]
    if USE_EARLY_STOPPING:
        trainer_callbacks.append(EarlyStoppingCallback(early_stopping_patience=10))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=trainer_callbacks
    )

    trainer.train()

    # -------- Persist training logs --------
    os.makedirs("../outputs", exist_ok=True)
    with open("../outputs/training_history.json", "w") as f:
        json.dump(log_callback.history, f, indent=2)
    print("Training history saved to ../outputs/training_history.json")

    # -------- Plot metrics + CSV --------
    def plot_training_metrics(callback,
                              save_path="../outputs/training_metrics.png",
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

        dfm = pd.DataFrame({
            "epoch": epochs,
            "val_loss": val_loss,
            "train_loss": train_loss,
            "val_accuracy": val_accuracy
        })
        dfm.to_csv(csv_path, index=False)
        print(f"CSV saved to {csv_path}")

    plot_training_metrics(log_callback)

    # -------- Save best model --------
    trainer.save_model("../outputs/best_model")

    # -------- Test set evaluation --------
    pred_output = trainer.predict(test_dataset)
    preds = np.argmax(pred_output.predictions, axis=1)
    labels = pred_output.label_ids

    target_names = [LABEL_NAME_BY_ID[0], LABEL_NAME_BY_ID[1], LABEL_NAME_BY_ID[2]]
    print("=== Classification Report ===")
    print(classification_report(labels, preds, target_names=target_names, digits=4))

    report_dict = classification_report(labels, preds, output_dict=True, target_names=target_names)

    # Append test row to training_metrics.csv
    test_metrics = {
        "epoch": "test",
        "val_loss": pred_output.metrics.get("test_loss", None),
        "train_loss": None,
        "val_accuracy": report_dict["accuracy"],
        "precision": report_dict["macro avg"]["precision"],
        "recall": report_dict["macro avg"]["recall"],
        "f1": report_dict["macro avg"]["f1-score"]
    }

    metrics_csv_path = "../outputs/training_metrics.csv"
    if os.path.exists(metrics_csv_path):
        df_metrics = pd.read_csv(metrics_csv_path)
    else:
        df_metrics = pd.DataFrame(columns=["epoch", "val_loss", "train_loss", "val_accuracy",
                                           "precision", "recall", "f1"])
    df_metrics = pd.concat([df_metrics, pd.DataFrame([test_metrics])], ignore_index=True)
    df_metrics.to_csv(metrics_csv_path, index=False)
    print(f"Test evaluation metrics appended to {metrics_csv_path}")

    with open("../outputs/test_classification_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)
    print("Full test classification report saved to ../outputs/test_classification_report.json")
