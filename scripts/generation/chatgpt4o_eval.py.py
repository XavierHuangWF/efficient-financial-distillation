import os
import re
import json
import time
import random
import argparse

import openai
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


SEED = 24266
SLEEP_TIME = 1.2
MODEL = "gpt-4o"


DATASET_CONFIG = {
    "phrasebank": {
        "load_fn": "load_phrasebank",
        "labels": {0: "Negative", 1: "Neutral", 2: "Positive"},
        "output_path": "./results/chatgpt4o_phrasebank_predictions.json",
        "report_path": "./results/chatgpt4o_phrasebank_classification_report.json",
    },
    "twitter": {
        "load_fn": "load_twitter",
        "labels": {0: "Bearish", 1: "Bullish", 2: "Neutral"},
        "output_path": "./results/chatgpt4o_twitter_predictions.json",
        "report_path": "./results/chatgpt4o_twitter_classification_report.json",
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_phrasebank(seed: int) -> pd.DataFrame:
    print("Loading Financial PhraseBank...")
    dataset = load_dataset("takala/financial_phrasebank", "sentences_allagree", split="train")
    df = pd.DataFrame(dataset)

    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=seed
    )

    return test_df.reset_index(drop=True)


def load_twitter(seed: int) -> pd.DataFrame:
    print("Loading Twitter Financial News Sentiment...")
    ds_train = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
    ds_val = load_dataset("zeroshot/twitter-financial-news-sentiment", split="validation")
    df = pd.concat([pd.DataFrame(ds_train), pd.DataFrame(ds_val)], ignore_index=True)

    if "text" in df.columns and "sentence" not in df.columns:
        df = df.rename(columns={"text": "sentence"})
    elif "sentence" not in df.columns:
        text_col = next((c for c in df.columns if df[c].dtype == object), None)
        if text_col is None:
            raise ValueError("Could not find a text column.")
        df = df.rename(columns={text_col: "sentence"})

    df["label"] = df["label"].astype(int)

    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=seed
    )

    return test_df.reset_index(drop=True)


def make_prompt(sentence: str, dataset: str) -> str:
    if dataset == "phrasebank":
        return f"""
You are a sentiment analysis assistant for financial news.
Classify the sentiment of the following financial news sentence into exactly one of:
- Negative
- Neutral
- Positive

Sentence: "{sentence}"

Respond with just one word: Negative, Neutral, or Positive.
""".strip()

    if dataset == "twitter":
        return f"""
You are a sentiment analysis assistant for financial news.
Classify the sentiment of the following sentence into exactly one of:
- Bearish
- Bullish
- Neutral

Sentence: "{sentence}"

Respond with just one word: Bearish, Bullish, or Neutral.
""".strip()

    raise ValueError(f"Unsupported dataset: {dataset}")


def parse_label(pred_text: str, dataset: str) -> int:
    t = pred_text.strip().lower()
    t_clean = re.sub(r"[^a-z]+", " ", t).strip()

    if dataset == "phrasebank":
        if "negative" in t_clean:
            return 0
        if "neutral" in t_clean:
            return 1
        if "positive" in t_clean:
            return 2
        return -1

    if dataset == "twitter":
        if "bearish" in t_clean:
            return 0
        if "bullish" in t_clean:
            return 1
        if "neutral" in t_clean:
            return 2
        if "negative" in t_clean or "sell" in t_clean or "pessim" in t_clean:
            return 0
        if "positive" in t_clean or "buy" in t_clean or "optim" in t_clean:
            return 1
        return -1

    raise ValueError(f"Unsupported dataset: {dataset}")


def evaluate_dataset(dataset: str, model: str, sleep_time: float, seed: int) -> None:
    if dataset not in DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset: {dataset}")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    set_seed(seed)
    config = DATASET_CONFIG[dataset]
    id2label = config["labels"]
    label2id = {v: k for k, v in id2label.items()}

    load_fn = globals()[config["load_fn"]]
    test_df = load_fn(seed=seed)

    client = openai.OpenAI(api_key=api_key)
    predictions = []

    print(f"Running inference on {dataset} test split...")

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        sentence = row["sentence"]
        true_label = int(row["label"])

        prompt = make_prompt(sentence, dataset)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )

            pred_text = response.choices[0].message.content.strip()
            pred_label = parse_label(pred_text, dataset)

            predictions.append(
                {
                    "sentence": sentence,
                    "true_label": id2label[true_label],
                    "pred_label": id2label.get(pred_label, "Unknown"),
                    "raw_response": pred_text,
                }
            )

        except Exception as e:
            print("Error:", e)
            predictions.append(
                {
                    "sentence": sentence,
                    "true_label": id2label[true_label],
                    "pred_label": "Error",
                    "raw_response": str(e),
                }
            )

        time.sleep(sleep_time)

    os.makedirs("./results", exist_ok=True)

    with open(config["output_path"], "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"Saved predictions to {config['output_path']}")

    valid = [p for p in predictions if p["pred_label"] in label2id]
    y_true = [label2id[p["true_label"]] for p in valid]
    y_pred = [label2id[p["pred_label"]] for p in valid]

    report = classification_report(
        y_true,
        y_pred,
        target_names=[id2label[i] for i in sorted(id2label.keys())],
        output_dict=True,
    )

    with open(config["report_path"], "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("Classification Report:")
    print(json.dumps(report, indent=2))
    print(f"Saved classification report to {config['report_path']}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["phrasebank", "twitter"], required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--sleep-time", type=float, default=SLEEP_TIME)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_dataset(
        dataset=args.dataset,
        model=args.model,
        sleep_time=args.sleep_time,
        seed=args.seed,
    )