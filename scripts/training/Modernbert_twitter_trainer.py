import os
import re
import json
import random
import argparse
import itertools

import numpy as np
import pandas as pd
import torch
import evaluate
import matplotlib.pyplot as plt

from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, classification_report
from sentence_transformers import SentenceTransformer

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)

from LogExportCallback import LogExportCallback


SEED = 24266
MODEL_NAME = "answerdotai/ModernBERT-base"
TEXT_CANDIDATES = ["sentence", "text", "tweet", "content", "document", "message", "headline"]
LABEL_NAME_BY_ID = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
LABEL_ID_BY_NAME = {"Bearish": 0, "Bullish": 1, "Neutral": 2}


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_text(text):
    if not isinstance(text, str):
        return text
    text = text.replace("\\", "\")
    try:
        text = bytes(text, "utf-8").decode("unicode_escape")
    except Exception:
        pass
    text = re.sub(r"^\d+\.\s*", "", text)
    text = text.replace('"', "")
    return text.strip()


def plot_training_metrics(callback, save_path, csv_path):
    history = callback.history
    if not history.get("epoch"):
        print("No training history found. Skipping plot export.")
        return

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

    df = pd.DataFrame({
        "epoch": epochs,
        "val_loss": val_loss,
        "train_loss": train_loss,
        "val_accuracy": val_accuracy,
    })
    df.to_csv(csv_path, index=False)
    print(f"CSV saved to {csv_path}")


def find_text_column(df: pd.DataFrame) -> str:
    for c in TEXT_CANDIDATES:
        if c in df.columns:
            return c
    for c in df.columns:
        if df[c].dtype == object:
            return c
    raise ValueError("Could not find a text column in dataset.")


def load_twitter_dataset() -> pd.DataFrame:
    ds_train = load_dataset(
        "zeroshot/twitter-financial-news-sentiment",
        split="train",
        trust_remote_code=True,
    )
    ds_val = load_dataset(
        "zeroshot/twitter-financial-news-sentiment",
        split="validation",
        trust_remote_code=True,
    )
    df = pd.concat([pd.DataFrame(ds_train), pd.DataFrame(ds_val)], ignore_index=True)

    if "label_text" in df.columns:
        lt = df["label_text"].astype(str).str.strip().str.title()
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

    text_col = find_text_column(df)
    if text_col != "sentence":
        df = df.rename(columns={text_col: "sentence"})

    df = df[["sentence", "label"]].copy()
    df["sentence"] = df["sentence"].astype(str).apply(clean_text)
    return df


def stratified_split(data: pd.DataFrame, seed: int):
    train_df, temp_df = train_test_split(
        data,
        test_size=0.2,
        stratify=data["label"],
        random_state=seed,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label"],
        random_state=seed,
    )

    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    for label, count in train_df["label"].value_counts().sort_index().items():
        pct = count / len(train_df) * 100
        print(f"Label {label} ({LABEL_NAME_BY_ID[label]}): {count} ({pct:.2f}%)")

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def generate_embeddings(sentences):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model.encode(sentences, show_progress_bar=True)


def cluster_embeddings(embeddings, num_clusters: int, seed: int):
    model = KMeans(n_clusters=num_clusters, random_state=seed, n_init=10)
    model.fit(embeddings)
    return model


def select_seed_data(df: pd.DataFrame, class_balanced: bool, samples_per_class: int, num_clusters: int, seed: int):
    if class_balanced:
        seed_frames = []
        for label in sorted(df["label"].unique()):
            subset = df[df["label"] == label].reset_index(drop=True)
            embeddings = generate_embeddings(subset["sentence"].tolist())
            cluster_model = cluster_embeddings(embeddings, num_clusters=samples_per_class, seed=seed)
            closest, _ = pairwise_distances_argmin_min(cluster_model.cluster_centers_, embeddings)
            selected = subset.iloc[closest]
            seed_frames.append(selected)
        seed_df = pd.concat(seed_frames, ignore_index=True)
    else:
        embeddings = generate_embeddings(df["sentence"].tolist())
        cluster_model = cluster_embeddings(embeddings, num_clusters=num_clusters, seed=seed)
        closest, _ = pairwise_distances_argmin_min(cluster_model.cluster_centers_, embeddings)
        seed_df = df.iloc[closest].reset_index(drop=True)

    return seed_df.drop_duplicates(subset=["sentence", "label"]).reset_index(drop=True)


def normalize_label_to_id(x):
    if isinstance(x, (int, np.integer)):
        return int(x) if x in (0, 1, 2) else np.nan
    if isinstance(x, float):
        if pd.isna(x):
            return np.nan
        xi = int(x)
        return xi if xi in (0, 1, 2) else np.nan

    if x is None:
        return np.nan

    y = str(x).strip().title()
    if not y:
        return np.nan
    if y == "Negative":
        y = "Bearish"
    elif y == "Positive":
        y = "Bullish"

    return LABEL_ID_BY_NAME.get(y, np.nan)


def load_jsonl_to_train_df(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    df = pd.DataFrame(rows)
    if not {"input", "output"}.issubset(df.columns):
        raise ValueError(f"{path} missing required keys 'input' and/or 'output'")

    df = df.rename(columns={"input": "sentence", "output": "label"})[["sentence", "label"]].copy()

    before = len(df)
    df = df[df["label"].notna()]
    df = df[df["label"].astype(str).str.strip().ne("")]
    dropped = before - len(df)
    if dropped:
        print(f"[WARN] Dropped {dropped} row(s) with missing/empty labels in {path}")

    df["sentence"] = df["sentence"].astype(str).apply(clean_text)
    df["label"] = df["label"].apply(normalize_label_to_id)

    bad = df["label"].isna().sum()
    if bad:
        print(f"[WARN] Dropping {bad} row(s) with unmappable labels in {path}")

    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
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
    print(df.head(3))


accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    prec = precision_metric.compute(predictions=predictions, references=labels, average="macro")
    rec = recall_metric.compute(predictions=predictions, references=labels, average="macro")
    f1_score = f1_metric.compute(predictions=predictions, references=labels, average="macro")

    return {
        "accuracy": acc["accuracy"],
        "precision": prec["precision"],
        "recall": rec["recall"],
        "f1": f1_score["f1"],
    }


def freeze_n_transformer_layers(model, n):
    if n <= 0:
        print("No transformer layers are frozen. Full fine-tuning enabled.")
        return

    candidates = [
        ("model.layers", lambda m: getattr(getattr(m, "model", m), "layers", None)),
        ("model.encoder.layers", lambda m: getattr(getattr(getattr(m, "model", m), "encoder", m), "layers", None)),
        ("encoder.layer", lambda m: getattr(getattr(m, "encoder", m), "layer", None)),
    ]

    for name, getter in candidates:
        layers = getter(model)
        if layers is not None:
            L = len(layers)
            k = min(n, L)
            for p in itertools.chain.from_iterable(layer.parameters() for layer in layers[:k]):
                p.requires_grad = False
            print(f"Freezing first {k}/{L} transformer layers via path '{name}'.")
            return

    print("[WARN] Could not locate transformer layers to freeze; proceeding without freezing.")


def build_train_val_test(args):
    full_data = load_twitter_dataset()
    train_df_full, val_df, test_df = stratified_split(full_data, seed=args.seed)

    if args.mode == "full":
        train_df = train_df_full.copy()
        print(f"Using full supervised train split: {len(train_df)} rows")

    elif args.mode == "seed":
        if args.seed_source == "cluster":
            train_df = select_seed_data(
                df=train_df_full,
                class_balanced=args.class_balanced,
                samples_per_class=args.samples_per_class,
                num_clusters=args.num_clusters,
                seed=args.seed,
            )
            print(f"Cluster-selected seed size: {len(train_df)}")
        elif args.seed_source == "jsonl":
            if not args.seed_jsonl:
                raise ValueError("--seed-jsonl is required when --seed-source jsonl")
            train_df = load_jsonl_to_train_df(args.seed_jsonl)
            print(f"Loaded seed JSONL size: {len(train_df)}")
        else:
            raise ValueError(f"Unsupported seed source: {args.seed_source}")

    elif args.mode == "synthetic":
        if not args.synthetic_jsonl:
            raise ValueError("--synthetic-jsonl is required when --mode synthetic")
        if not args.seed_jsonl:
            raise ValueError("--seed-jsonl is required when --mode synthetic")

        syn_df = load_jsonl_to_train_df(args.synthetic_jsonl)
        seed_df = load_jsonl_to_train_df(args.seed_jsonl)

        print(f"Synthetic training size (pre-merge): {len(syn_df)}")
        print(f"Seed size: {len(seed_df)}")

        train_df = pd.concat([syn_df, seed_df], ignore_index=True)
        before = len(train_df)
        train_df = train_df.drop_duplicates(subset=["sentence", "label"]).reset_index(drop=True)
        after = len(train_df)
        if after != before:
            print(f"Deduped {before - after} duplicate rows (sentence+label).")
        print(f"Synthetic + seed training size: {len(train_df)}")

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    preflight_report(train_df, "train_df")
    preflight_report(val_df, "val_df")
    preflight_report(test_df, "test_df")

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified ModernBERT Twitter Financial News trainer")

    parser.add_argument("--mode", choices=["full", "seed", "synthetic"], required=True)
    parser.add_argument("--seed-source", choices=["cluster", "jsonl"], default="jsonl")

    parser.add_argument("--seed-jsonl", type=str, default=None)
    parser.add_argument("--synthetic-jsonl", type=str, default=None)

    parser.add_argument("--class-balanced", action="store_true")
    parser.add_argument("--samples-per-class", type=int, default=35)
    parser.add_argument("--num-clusters", type=int, default=105)

    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--freeze-layers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.1)

    parser.add_argument("--use-early-stopping", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=10)

    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--output-dir", type=str, default="../outputs")
    parser.add_argument("--seed", type=int, default=SEED)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    if torch.cuda.is_available():
        print("GPU device:", torch.cuda.get_device_name(0))

    train_df, val_df, test_df = build_train_val_test(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    def tokenize_function(batch):
        return tokenizer(batch["sentence"], truncation=True)

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        trust_remote_code=True,
    )

    freeze_n_transformer_layers(model, args.freeze_layers)

    training_args = TrainingArguments(
        seed=args.seed,
        data_seed=args.seed,
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        label_smoothing_factor=args.label_smoothing,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_strategy="epoch",
        save_total_limit=1,
    )

    log_callback = LogExportCallback()
    trainer_callbacks = [log_callback]
    if args.use_early_stopping:
        trainer_callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=trainer_callbacks,
    )

    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)

    training_history_path = os.path.join(args.output_dir, "training_history.json")
    with open(training_history_path, "w", encoding="utf-8") as f:
        json.dump(log_callback.history, f, indent=2)
    print(f"Training history saved to {training_history_path}")

    metrics_plot_path = os.path.join(args.output_dir, "training_metrics.png")
    metrics_csv_path = os.path.join(args.output_dir, "training_metrics.csv")
    plot_training_metrics(log_callback, save_path=metrics_plot_path, csv_path=metrics_csv_path)

    best_model_path = os.path.join(args.output_dir, "best_model")
    trainer.save_model(best_model_path)
    print(f"Best model saved to {best_model_path}")

    pred_output = trainer.predict(test_dataset)
    preds = np.argmax(pred_output.predictions, axis=1)
    labels = pred_output.label_ids

    target_names = [LABEL_NAME_BY_ID[0], LABEL_NAME_BY_ID[1], LABEL_NAME_BY_ID[2]]
    report = classification_report(
        labels,
        preds,
        output_dict=True,
        target_names=target_names,
    )

    print("=== Classification Report ===")
    print(classification_report(
        labels,
        preds,
        target_names=target_names,
        digits=4,
    ))

    test_metrics = {
        "epoch": "test",
        "val_loss": pred_output.metrics.get("test_loss"),
        "train_loss": None,
        "val_accuracy": report["accuracy"],
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1": report["macro avg"]["f1-score"],
    }

    if os.path.exists(metrics_csv_path):
        df_metrics = pd.read_csv(metrics_csv_path)
        df_metrics = pd.concat([df_metrics, pd.DataFrame([test_metrics])], ignore_index=True)
    else:
        df_metrics = pd.DataFrame([test_metrics])

    df_metrics.to_csv(metrics_csv_path, index=False)
    print(f"Test evaluation metrics saved to {metrics_csv_path}")

    test_report_path = os.path.join(args.output_dir, "test_classification_report.json")
    with open(test_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Full test classification report saved to {test_report_path}")
