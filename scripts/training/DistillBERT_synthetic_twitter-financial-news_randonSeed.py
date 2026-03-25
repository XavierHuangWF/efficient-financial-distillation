import os
import re
import json
import random
import pandas as pd
import numpy as np
import torch
import evaluate
import matplotlib.pyplot as plt

from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)

# =======================
# Reproducibility
# =======================
def set_seed(seed=24266):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(24266)

# =======================
# Config
# =======================
NUM_EPOCHS = 50
FREEZE_LAYERS = 4
USE_SYNTHETIC_DATA =   True
USE_SEED_ONLY =  False
USE_EARLY_STOPPING = True

SYNTHETIC_PATH = "../outputs/2synthetic_data_from_Seed_random.jsonl"
SEED_PATH = "../outputs/2seed_data_twitter_random.jsonl"
OUTPUT_DIR = "../outputs"

# ---------- helpers ----------
TEXT_CANDIDATES = ["sentence", "text", "tweet", "content", "document", "message", "headline"]

LABEL_NAME_BY_ID = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
LABEL_ID_BY_NAME = {"Bearish": 0, "Bullish": 1, "Neutral": 2}

# =======================
# Cleaning
# =======================
def clean_text(text):
    if not isinstance(text, str):
        return text
    text = text.replace('\\\\', '\\')
    try:
        text = bytes(text, "utf-8").decode("unicode_escape")
    except Exception:
        pass
    text = re.sub(r"^\d+\.\s*", "", text)  # remove leading enumerations like "12. "
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

# =======================
# HF dataset loader (train + val merged; we do our own split)
# =======================
def DataLoad():
    ds_train = load_dataset(
        "zeroshot/twitter-financial-news-sentiment",
        split="train",
        trust_remote_code=True
    )
    ds_val = load_dataset(
        "zeroshot/twitter-financial-news-sentiment",
        split="validation",
        trust_remote_code=True
    )
    df = pd.concat([pd.DataFrame(ds_train), pd.DataFrame(ds_val)], ignore_index=True)

    # Prefer explicit label text when available
    if "label_text" in df.columns:
        lt = df["label_text"].astype(str).str.strip().str.title()
        # ensure Negative/Positive synonyms map the same way as the other script
        lt = lt.replace({"Negative": "Bearish", "Positive": "Bullish"})
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

# =======================
# Robust label normalizer for JSONL (matches the other script’s behavior)
# =======================
def _normalize_label_to_id(x):
    """
    Accept:
      - Strings: Bearish/Bullish/Neutral (case/space tolerant),
                 or Negative->Bearish, Positive->Bullish
      - Numerics: 0/1/2
    Return 0/1/2, or np.nan for unmappable (the caller drops those).
    """
    # numeric 0/1/2
    if isinstance(x, (int, np.integer)):
        return int(x) if x in (0, 1, 2) else np.nan
    if isinstance(x, float):
        if pd.isna(x):
            return np.nan
        xi = int(x)
        return xi if xi in (0, 1, 2) else np.nan

    if not isinstance(x, str):
        return np.nan

    y = x.strip().title()
    if not y:
        return np.nan
    if y == "Negative":
        y = "Bearish"
    elif y == "Positive":
        y = "Bullish"
    return LABEL_ID_BY_NAME.get(y, np.nan)

def _load_jsonl_to_train_df(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame(rows)

    if not {"input", "output"}.issubset(df.columns):
        raise ValueError(f"{path} missing required keys 'input' and/or 'output'")

    df = df.rename(columns={"input": "sentence", "output": "label"})[["sentence", "label"]].copy()

    # Drop NaN and blank labels BEFORE mapping (matches the other script)
    before = len(df)
    df = df[df["label"].notna()]
    df = df[df["label"].astype(str).str.strip().ne("")]
    dropped = before - len(df)
    if dropped:
        print(f"[WARN] Dropped {dropped} row(s) with missing/empty labels in {path}")

    # Clean text
    df["sentence"] = df["sentence"].astype(str).apply(clean_text)

    # Map labels; drop unmappable instead of raising
    df["label"] = df["label"].apply(_normalize_label_to_id)
    bad = df["label"].isna().sum()
    if bad:
        print(f"[WARN] Dropping {bad} row(s) with unmappable labels in {path}")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # Drop empty sentences AFTER cleaning
    df = df[df["sentence"].str.strip().ne("")].reset_index(drop=True)
    return df

def preflight_report(df: pd.DataFrame, name: str):
    print(f"\n=== Preflight: {name} ===")
    print(f"Rows: {len(df)}")
    vc = df["label"].value_counts().sort_index()
    for k, v in vc.items():
        print(f"  {k} ({LABEL_NAME_BY_ID[k]}): {v}")
    empty_sents = (df["sentence"].str.strip() == "").sum()
    if empty_sents:
        print(f"  [WARN] Empty sentences: {empty_sents}")
    print("Sample:")
    print(df.head(3))

# =======================
# Metrics / Tokenization
# =======================
TOKENIZER = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize_function(batch):
    return TOKENIZER(batch["sentence"], truncation=True)

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
        "f1": f1_score["f1"],
    }

# =======================
# Main
# =======================
if __name__ == "__main__":
    if torch.cuda.is_available():
        print("GPU device:", torch.cuda.get_device_name(0))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ----------- DATA SELECTION -----------
    if USE_SYNTHETIC_DATA:
        syn_df = _load_jsonl_to_train_df(SYNTHETIC_PATH)
        print(f"Synthetic training rows: {len(syn_df)}")

        if os.path.exists(SEED_PATH):
            seed_df = _load_jsonl_to_train_df(SEED_PATH)
            print(f"Seed rows: {len(seed_df)}")
            train_df = pd.concat([syn_df, seed_df], ignore_index=True)
            # De-dup exact (sentence, label)
            before = len(train_df)
            train_df = train_df.drop_duplicates(subset=["sentence", "label"]).reset_index(drop=True)
            after = len(train_df)
            if after != before:
                print(f"Deduped {before - after} duplicate rows (sentence+label).")
        else:
            print(f"[WARN] Seed file not found at {SEED_PATH}; training on synthetic only.")
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

    preflight_report(train_df, "train_df")
    preflight_report(val_df, "val_df")
    preflight_report(test_df, "test_df")

    # ----------- HF DATASETS -----------
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset   = Dataset.from_pandas(val_df.reset_index(drop=True))
    test_dataset  = Dataset.from_pandas(test_df.reset_index(drop=True))

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset   = val_dataset.map(tokenize_function, batched=True)
    test_dataset  = test_dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=TOKENIZER)

    # ----------- Model -----------
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=3
    )

    if FREEZE_LAYERS > 0:
        total_layers = len(model.distilbert.transformer.layer)  # clamp for safety
        freeze_upto = min(FREEZE_LAYERS, total_layers)
        print(f"Freezing first {freeze_upto}/{total_layers} transformer layers...")
        for param in model.distilbert.transformer.layer[:freeze_upto].parameters():
            param.requires_grad = False
    else:
        print("No transformer layers are frozen. Full fine-tuning enabled.")

    # ----------- Training Args -----------
    training_args = TrainingArguments(
        seed=24266,
        data_seed=24266,
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",   # correct key; safe to keep
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
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_strategy="epoch",
        save_total_limit=1,
    )

    # ----------- Trainer -----------
    from LogExportCallback import LogExportCallback
    log_callback = LogExportCallback()
    trainer_callbacks = [log_callback]
    if USE_EARLY_STOPPING:
        trainer_callbacks.append(EarlyStoppingCallback(early_stopping_patience=10))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=TOKENIZER,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=trainer_callbacks,
    )

    trainer.train()

    # ----------- Export logs/plots -----------
    with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w") as f:
        json.dump(log_callback.history, f, indent=2)
    print(f"Training history saved to {os.path.join(OUTPUT_DIR, 'training_history.json')}")

    def plot_training_metrics(callback,
                              save_path=os.path.join(OUTPUT_DIR, "training_metrics.png"),
                              csv_path=os.path.join(OUTPUT_DIR, "training_metrics.csv")):
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
        # plt.show()  # optional

        dfm = pd.DataFrame({
            "epoch": epochs,
            "val_loss": val_loss,
            "train_loss": train_loss,
            "val_accuracy": val_accuracy
        })
        dfm.to_csv(csv_path, index=False)
        print(f"CSV saved to {csv_path}")

    plot_training_metrics(log_callback)

    # ----------- Save model -----------
    best_model_dir = os.path.join(OUTPUT_DIR, "best_model")
    trainer.save_model(best_model_dir)

    # ----------- Final test evaluation -----------
    pred_output = trainer.predict(test_dataset)
    preds = np.argmax(pred_output.predictions, axis=1)
    labels = pred_output.label_ids

    target_names = [LABEL_NAME_BY_ID[0], LABEL_NAME_BY_ID[1], LABEL_NAME_BY_ID[2]]
    report = classification_report(labels, preds, output_dict=True, target_names=target_names)

    print("=== Classification Report ===")
    print(classification_report(labels, preds, target_names=target_names, digits=4))

    # append a final "test" row to training_metrics.csv
    test_metrics = {
        "epoch": "test",
        "val_loss": pred_output.metrics.get("test_loss"),
        "train_loss": None,
        "val_accuracy": report["accuracy"],
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1": report["macro avg"]["f1-score"],
    }

    tm_csv = os.path.join(OUTPUT_DIR, "training_metrics.csv")
    df_metrics = pd.read_csv(tm_csv) if os.path.exists(tm_csv) else pd.DataFrame(
        columns=["epoch","val_loss","train_loss","val_accuracy"]
    )
    df_metrics = pd.concat([df_metrics, pd.DataFrame([test_metrics])], ignore_index=True)
    df_metrics.to_csv(tm_csv, index=False)
    print(f"Test evaluation metrics appended to {tm_csv}")

    with open(os.path.join(OUTPUT_DIR, "test_classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"Full test classification report saved to {os.path.join(OUTPUT_DIR, 'test_classification_report.json')}")
