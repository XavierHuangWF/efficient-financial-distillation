"""
FinBERT baseline evaluation on:
  - Financial PhraseBank (sentences_allagree), OR
  - Twitter Financial News Sentiment (zeroshot/twitter-financial-news-sentiment)

using the SAME stratified 80/10/10 split (random_state=24266).

Pipeline:
- Load selected dataset
- Create a stratified 80/10/10 split with random_state=24266
- Run OFF-THE-SHELF FinBERT (ProsusAI/finbert) inference on the test split (no fine-tuning)
- Compute Accuracy + Macro Precision/Recall/F1
- Print a classification report
- Save metrics + report JSON to ../outputs/

Requirements:
  pip install -U transformers datasets evaluate torch scikit-learn
"""

import os
import json
import random
import numpy as np
import torch
import evaluate
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------
# Config
# -----------------------
SEED = 24266
MODEL_NAME = "ProsusAI/finbert"

# Switch dataset here:
#   "phrasebank_allagree" -> takala/financial_phrasebank (sentences_allagree)
#   "twitter_financial_news" -> zeroshot/twitter-financial-news-sentiment (train+validation combined)
EVAL_DATASET = "twitter_financial_news"  # <-- SWITCH HERE

BATCH_SIZE = 64
MAX_LENGTH = 512  # match your training script token length

# PhraseBank metadata
PHRASEBANK_DATASET_NAME = "takala/financial_phrasebank"
PHRASEBANK_CONFIG = "sentences_allagree"
PHRASEBANK_TEXT_COL = "sentence"
PHRASEBANK_LABEL_COL = "label"
PHRASEBANK_TARGET_NAMES = ["Negative", "Neutral", "Positive"]
PHRASEBANK_LABEL_STR2ID = {"negative": 0, "neutral": 1, "positive": 2}

# Twitter Financial News metadata
TWITTER_DATASET_NAME = "zeroshot/twitter-financial-news-sentiment"
TWITTER_TEXT_COL = "sentence"
TWITTER_LABEL_COL = "label"
TWITTER_TARGET_NAMES = ["Bearish", "Bullish", "Neutral"]
# FinBERT outputs: negative/positive/neutral
# Twitter ids: 0=Bearish, 1=Bullish, 2=Neutral
TWITTER_LABEL_STR2ID = {"negative": 0, "positive": 1, "neutral": 2}


# -----------------------
# Helpers
# -----------------------
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    return device


def stratified_split_indices(labels: np.ndarray, seed: int = SEED):
    """
    Exact split logic used in your Trainer code:
    1) train vs temp: 80/20
    2) val vs test from temp: 50/50 -> 10/10 overall
    """
    idx = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        idx, test_size=0.2, stratify=labels, random_state=seed
    )
    temp_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_labels, random_state=seed
    )
    return train_idx, val_idx, test_idx


def build_label_remap(model, label_str2id: dict) -> dict:
    """
    Maps the model's output label IDs -> dataset label IDs, based on model.config.id2label strings.

    Examples:
      PhraseBank expects: negative->0, neutral->1, positive->2
      Twitter expects:    negative->0(Bearish), positive->1(Bullish), neutral->2(Neutral)
    """
    id2label = model.config.id2label  # e.g. {0:'negative',1:'neutral',2:'positive'}
    remap = {}
    for i, lbl in id2label.items():
        key = str(lbl).lower().strip()
        if key not in label_str2id:
            raise ValueError(f"Unexpected model id2label '{lbl}' for id={i}. Expected one of {list(label_str2id.keys())}.")
        remap[int(i)] = int(label_str2id[key])
    return remap


def batched_predict(model, tokenizer, texts, device, batch_size=BATCH_SIZE, max_length=MAX_LENGTH):
    model.eval()
    preds = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits

        preds.extend(logits.argmax(dim=-1).detach().cpu().numpy().tolist())

    return np.array(preds, dtype=int)


def load_eval_dataset(eval_dataset: str):
    """
    Returns:
      texts: List[str]
      labels: np.ndarray[int]
      dataset_id: str (for metrics file naming)
      target_names: List[str]
      label_str2id: dict[str,int] (for remap)
    """
    if eval_dataset == "phrasebank_allagree":
        ds = load_dataset(PHRASEBANK_DATASET_NAME, PHRASEBANK_CONFIG, split="train", trust_remote_code=True)
        texts = [ex[PHRASEBANK_TEXT_COL] for ex in ds]
        labels = np.array(ds[PHRASEBANK_LABEL_COL], dtype=int)
        dataset_id = f"{PHRASEBANK_DATASET_NAME}/{PHRASEBANK_CONFIG}"
        return texts, labels, dataset_id, PHRASEBANK_TARGET_NAMES, PHRASEBANK_LABEL_STR2ID

    if eval_dataset == "twitter_financial_news":
        ds_train = load_dataset(TWITTER_DATASET_NAME, split="train", trust_remote_code=True)
        ds_val = load_dataset(TWITTER_DATASET_NAME, split="validation", trust_remote_code=True)

        # Combine train + validation, match your other script's split setup
        texts = []
        labels = []

        # Prefer label_text if present; otherwise use numeric label.
        def _consume(ds_part):
            nonlocal texts, labels
            cols = ds_part.column_names
            has_label_text = "label_text" in cols
            has_label = "label" in cols

            for ex in ds_part:
                texts.append(ex.get(TWITTER_TEXT_COL, ex.get("text", ex.get("tweet", ex.get("content", "")))))
                if has_label_text:
                    lt = str(ex["label_text"]).strip().lower()
                    # Expect bearish/bullish/neutral; map to 0/1/2
                    if lt == "bearish":
                        labels.append(0)
                    elif lt == "bullish":
                        labels.append(1)
                    elif lt == "neutral":
                        labels.append(2)
                    else:
                        raise ValueError(f"Unexpected label_text: {ex['label_text']}")
                else:
                    if not has_label:
                        raise ValueError("Twitter dataset missing 'label' and 'label_text'.")
                    y = int(ex["label"])
                    if y not in (0, 1, 2):
                        raise ValueError(f"Unexpected numeric label: {y}")
                    labels.append(y)

        _consume(ds_train)
        _consume(ds_val)

        labels = np.array(labels, dtype=int)
        dataset_id = f"{TWITTER_DATASET_NAME}/train+validation"
        return texts, labels, dataset_id, TWITTER_TARGET_NAMES, TWITTER_LABEL_STR2ID

    raise ValueError("EVAL_DATASET must be 'phrasebank_allagree' or 'twitter_financial_news'.")


# -----------------------
# Main
# -----------------------
def main():
    set_seed(SEED)
    device = get_device()

    texts_all, y_all, dataset_id, target_names, label_str2id = load_eval_dataset(EVAL_DATASET)

    _, _, test_idx = stratified_split_indices(y_all, seed=SEED)
    test_texts = [texts_all[int(i)] for i in test_idx]
    y_true = np.array([y_all[int(i)] for i in test_idx], dtype=int)

    print("Eval dataset:", EVAL_DATASET)
    print("Dataset id:", dataset_id)
    print("Test size:", len(test_texts))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, use_safetensors=True).to(device)

    # Remap FinBERT output ids -> dataset ids (PhraseBank or Twitter)
    remap = build_label_remap(model, label_str2id)

    raw_pred = batched_predict(model, tokenizer, test_texts, device=device)
    y_pred = np.array([remap[int(i)] for i in raw_pred], dtype=int)

    # Metrics
    acc_metric = evaluate.load("accuracy")
    prec_metric = evaluate.load("precision")
    rec_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    metrics = {
        "model": MODEL_NAME,
        "eval_dataset": EVAL_DATASET,
        "dataset": dataset_id,
        "seed": SEED,
        "split": "stratified 80/10/10 with random_state=24266; evaluate on test only",
        "n_test": int(len(y_true)),
        "accuracy": acc_metric.compute(predictions=y_pred, references=y_true)["accuracy"],
        "macro_precision": prec_metric.compute(predictions=y_pred, references=y_true, average="macro")["precision"],
        "macro_recall": rec_metric.compute(predictions=y_pred, references=y_true, average="macro")["recall"],
        "macro_f1": f1_metric.compute(predictions=y_pred, references=y_true, average="macro")["f1"],
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "device": str(device),
    }

    print("\n=== FinBERT Test Metrics ===")
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Macro-Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro-Recall:    {metrics['macro_recall']:.4f}")
    print(f"Macro-F1:        {metrics['macro_f1']:.4f}")

    # Classification report
    report_text = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    report_dict = classification_report(y_true, y_pred, target_names=target_names, digits=4, output_dict=True)

    print("\n=== Classification Report ===")
    print(report_text)

    # Save
    os.makedirs("../outputs", exist_ok=True)
    tag = "phrasebank" if EVAL_DATASET == "phrasebank_allagree" else "twitter"

    with open(f"../outputs/finbert_{tag}_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(f"../outputs/finbert_{tag}_test_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    print("\nSaved:")
    print(f" - ../outputs/finbert_{tag}_test_metrics.json")
    print(f" - ../outputs/finbert_{tag}_test_classification_report.json")


if __name__ == "__main__":
    main()