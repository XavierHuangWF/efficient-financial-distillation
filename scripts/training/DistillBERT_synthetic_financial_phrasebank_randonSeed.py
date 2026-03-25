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
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
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
#  2) Seed only
#  3) Full supervised (PhraseBank)  [when both switches below are False]

NUM_EPOCHS = 50
FREEZE_LAYERS = 4
USE_EARLY_STOPPING = True

# --- data regime switches (match Code 2) ---
USE_SYNTHETIC_DATA = False      # <- same as Code 2
USE_SEED_ONLY      = True       # <- same as Code 2

# --- paths (the ONLY intended difference vs Code 2 is SEED_PATH) ---
SYNTHETIC_PATH = "../outputs/2synthetic_data_from_randonSeed.jsonl"
SEED_PATH      = "../outputs/2seed_data_random.jsonl"   # <- random seed file here

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
    for label, count in train_df["label"].value_counts().items():
        percentage = count / len(train_df) * 100
        print(f"Label {label}: {count} ({percentage:.2f}%)")
    return train_df, val_df, test_df

# ===== Label normalization like Code 2 (title-cased names) =====
def _normalize_label_to_id(lbl: str) -> int:
    if not isinstance(lbl, str):
        raise ValueError(f"Label not a string: {lbl}")
    t = lbl.strip().title()
    if t in ["Negative", "Bearish"]:
        return 0
    if t in ["Neutral"]:
        return 1
    if t in ["Positive", "Bullish"]:
        return 2
    raise ValueError(f"Unexpected label string: {lbl}")

def _load_jsonl_as_df(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame(rows)
    if not {"input", "output"}.issubset(df.columns):
        raise ValueError(f"{path} missing required keys 'input' and/or 'output'")
    df = df.rename(columns={"input": "sentence", "output": "label"})[["sentence", "label"]].copy()
    df["sentence"] = df["sentence"].astype(str).apply(clean_text)
    df["label"] = df["label"].apply(_normalize_label_to_id).astype(int)
    return df

# ===================== Tokenizer / Metrics =====================
def tokenize_function(example):
    # Match Code 2: instantiate inside the function
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    return tokenizer(example["sentence"], truncation=True)

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

# ===================== MAIN =====================
if __name__ == "__main__":
    if torch.cuda.is_available():
        print("GPU device:", torch.cuda.get_device_name(0))

    if USE_SYNTHETIC_DATA:
        # Load synthetic
        syn_df = _load_jsonl_as_df(SYNTHETIC_PATH)
        print(f"Synthetic training size (pre-merge): {len(syn_df)}")

        # Also load seed and MERGE so we train on BOTH
        if os.path.exists(SEED_PATH):
            seed_df = _load_jsonl_as_df(SEED_PATH)
            print(f"Seed size: {len(seed_df)}")
            train_df = pd.concat([syn_df, seed_df], ignore_index=True)

            # Drop exact duplicates on (sentence, label)
            before = len(train_df)
            train_df = train_df.drop_duplicates(subset=["sentence", "label"]).reset_index(drop=True)
            after = len(train_df)
            if after != before:
                print(f"Deduped {before - after} duplicate rows (sentence+label).")
        else:
            print(f"Warning: seed file not found at {SEED_PATH}; training on synthetic only.")
            train_df = syn_df

        # Build val/test from PhraseBank (unchanged)
        full_data = DataLoad()
        _, val_df, test_df = Stratified_Split(full_data)

    elif USE_SEED_ONLY:
        train_df = _load_jsonl_as_df(SEED_PATH)
        print(f"Seed-only training size: {len(train_df)}")
        full_data = DataLoad()
        _, val_df, test_df = Stratified_Split(full_data)

    else:
        full_data = DataLoad()
        train_df, val_df, test_df = Stratified_Split(full_data)
        train_df["sentence"] = train_df["sentence"].apply(clean_text)
        val_df["sentence"] = val_df["sentence"].apply(clean_text)
        test_df["sentence"] = test_df["sentence"].apply(clean_text)

    # Convert to HF datasets
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset   = Dataset.from_pandas(val_df.reset_index(drop=True))
    test_dataset  = Dataset.from_pandas(test_df.reset_index(drop=True))

    # Tokenize
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset   = val_dataset.map(tokenize_function, batched=True)
    test_dataset  = test_dataset.map(tokenize_function, batched=True)

    # Training args (match Code 2)
    training_args = TrainingArguments(
        seed=SEED,
        data_seed=SEED,
        output_dir="../outputs",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=16,   # match Code 2
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

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=3
    )

    if FREEZE_LAYERS > 0:
        print(f"Freezing first {FREEZE_LAYERS} transformer layers...")
        for param in model.distilbert.transformer.layer[:FREEZE_LAYERS].parameters():
            param.requires_grad = False
    else:
        print("No transformer layers are frozen. Full fine-tuning enabled.")

    # NOTE: tokenizer is instantiated inside tokenize_function (as in Code 2)
    data_collator = DataCollatorWithPadding(tokenizer=DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased"))
    log_callback = LogExportCallback()

    trainer_callbacks = [log_callback]
    if USE_EARLY_STOPPING:
        trainer_callbacks.append(EarlyStoppingCallback(early_stopping_patience=10))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased"),
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=trainer_callbacks
    )

    trainer.train()

    with open("../outputs/training_history.json", "w") as f:
        json.dump(log_callback.history, f, indent=2)
    print("Training history saved to ../outputs/training_history.json")

    plot_training_metrics(log_callback)

    trainer.save_model("../outputs/best_model")

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

    # Match Code 2's CSV behavior (expects file to exist)
    df_metrics = pd.read_csv("../outputs/training_metrics.csv")
    df_metrics = pd.concat([df_metrics, pd.DataFrame([test_metrics])], ignore_index=True)
    df_metrics.to_csv("../outputs/training_metrics.csv", index=False)
    print("Test evaluation metrics appended to ../outputs/training_metrics.csv")

    with open("../outputs/test_classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Full test classification report saved to ../outputs/test_classification_report.json")
