import os
import pandas as pd
import torch
import numpy as np
import evaluate
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
import json
from LogExportCallback import LogExportCallback

from sentence_transformers import SentenceTransformer
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans
from transformers import EarlyStoppingCallback

# switched to Auto* for compatibility with ModernBERT
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

import random
def set_seed(seed=24266):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()

# ===== Model switch here =====
MODEL_NAME = "answerdotai/ModernBERT-base"

# 1. Whole training set
# NUM_EPOCHS = 20
# FREEZE_LAYERS = 0
# USE_SEED_DATA = False
# USE_EARLY_STOPPING = False

# 4. 105 Seed data + Frozen 4 layer + +early stop + class_balance
NUM_EPOCHS = 50
FREEZE_LAYERS = 4  # will clamp to the model's actual depth
USE_SEED_DATA = True
USE_CLASS_BALANCED_CLUSTERING = True
SAMPLES_PER_CLASS = 35
USE_EARLY_STOPPING = True

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

# === UPDATED ===
def DataLoad():
    # Load both provided splits and combine, then keep your own stratified split downstream.
    ds_train = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
    ds_val = load_dataset("zeroshot/twitter-financial-news-sentiment", split="validation")
    df = pd.concat([pd.DataFrame(ds_train), pd.DataFrame(ds_val)], ignore_index=True)
    # Dataset uses 'text' + integer 'label'. Rename to keep the rest of your pipeline identical.
    df = df.rename(columns={"text": "sentence"})
    # Ensure labels are ints 0/1/2 (0=Bearish, 1=Bullish, 2=Neutral)
    df["label"] = df["label"].astype(int)
    return df
# ==============

def Stratified_Split(data):
    train_df, temp_df = train_test_split(data, test_size=0.2, stratify=data["label"], random_state=24266)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=24266)
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    for label, count in train_df["label"].value_counts().items():
        percentage = count / len(train_df) * 100
        print(f"Label {label}: {count} ({percentage:.2f}%)")
    return train_df, val_df, test_df

def Generate_Embeddings(sentences):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model.encode(sentences, show_progress_bar=True)

def Cluster_Embeddings(embeddings, num_clusters=100):
    model = KMeans(n_clusters=num_clusters, random_state=24266)
    model.fit(embeddings)
    return model

def Select_Seed_Data(df):
    if USE_CLASS_BALANCED_CLUSTERING:
        seed_frames = []
        for label in sorted(df['label'].unique()):
            subset = df[df['label'] == label].reset_index(drop=True)
            embeddings = Generate_Embeddings(subset["sentence"].tolist())
            cluster_model = Cluster_Embeddings(embeddings, num_clusters=SAMPLES_PER_CLASS)
            closest, _ = pairwise_distances_argmin_min(cluster_model.cluster_centers_, embeddings)
            selected = subset.iloc[closest]
            seed_frames.append(selected)
        seed_df = pd.concat(seed_frames).reset_index(drop=True)
    else:
        embeddings = Generate_Embeddings(df["sentence"].tolist())
        clustering_model = Cluster_Embeddings(embeddings, num_clusters=105)
        closest, _ = pairwise_distances_argmin_min(clustering_model.cluster_centers_, embeddings)
        seed_df = df.iloc[closest].reset_index(drop=True)
    return seed_df

# tokenizer now uses AutoTokenizer with MODEL_NAME
def tokenize_function(example):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
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


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("GPU device:", torch.cuda.get_device_name(0))

    data = DataLoad()
    full_train_df, val_df, test_df = Stratified_Split(data)

    if USE_SEED_DATA:
        seed_df = Select_Seed_Data(full_train_df)
        print(f"Seed training size: {len(seed_df)}")
        train_dataset = Dataset.from_pandas(seed_df)
    else:
        print(f"Using full training data: {len(full_train_df)} samples")
        train_dataset = Dataset.from_pandas(full_train_df.reset_index(drop=True))

    val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        seed=24266,
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

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        use_safetensors=True
    )

    # ==== ONLY CHANGE: simple ModernBERT freeze ====
    if FREEZE_LAYERS > 0:
        print(f"Freezing first {FREEZE_LAYERS} transformer layers...")
        for param in model.model.layers[:FREEZE_LAYERS].parameters():
            param.requires_grad = False
    else:
        print("No transformer layers are frozen. Full fine-tuning enabled.")
    # ===============================================

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
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

    with open("../outputs/training_history.json", "w") as f:
        json.dump(log_callback.history, f, indent=2)
    print("Training history saved to ../outputs/training_history.json")

    plot_training_metrics(log_callback)

    trainer.save_model("../outputs/best_model")

    from sklearn.metrics import classification_report

    pred_output = trainer.predict(test_dataset)
    preds = np.argmax(pred_output.predictions, axis=1)
    labels = pred_output.label_ids

    report = classification_report(labels, preds, output_dict=True,
                                   target_names=["Bearish", "Bullish", "Neutral"])
    print("=== Classification Report ===")
    print(classification_report(labels, preds, target_names=["Bearish", "Bullish", "Neutral"], digits=4))

    test_metrics = {
        "epoch": "test",
        "val_loss": pred_output.metrics["test_loss"],
        "train_loss": None,
        "val_accuracy": report["accuracy"],
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1": report["macro avg"]["f1-score"]
    }

    df_metrics = pd.read_csv("../outputs/training_metrics.csv")
    df_metrics = pd.concat([df_metrics, pd.DataFrame([test_metrics])], ignore_index=True)
    df_metrics.to_csv("../outputs/training_metrics.csv", index=False)
    print("Test evaluation metrics appended to ../outputs/training_metrics.csv")

    with open("../outputs/test_classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Full test classification report saved to ../outputs/test_classification_report.json")
