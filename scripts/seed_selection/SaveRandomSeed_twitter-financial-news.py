import os
import pandas as pd
import numpy as np
import json
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# =======================
# Configuration
# =======================
SAMPLES_PER_CLASS = 35                # random samples per class
USE_CLASS_BALANCED_SAMPLING = True    # if False, will draw a flat total of 105
SEED = 24266  #24266 #26413 #36273
SAVE_PATH = "../../outputs/seed_data_twitter_random.jsonl"

# In case the text column name varies
TEXT_CANDIDATES = ["sentence", "text", "tweet", "content", "document", "message", "headline"]

def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)

def _find_text_column(df: pd.DataFrame) -> str:
    for c in TEXT_CANDIDATES:
        if c in df.columns:
            return c
    # fallback: first object dtype column
    for c in df.columns:
        if df[c].dtype == object:
            return c
    raise ValueError("Could not find a text column in dataset.")

def load_data():
    """
    Load zeroshot/twitter-financial-news-sentiment and combine 'train' + 'validation',
    then we will do our own stratified split (80/10/10) to avoid leakage.
    """
    ds_train = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train", trust_remote_code=True)
    ds_val   = load_dataset("zeroshot/twitter-financial-news-sentiment", split="validation", trust_remote_code=True)

    df = pd.concat([pd.DataFrame(ds_train), pd.DataFrame(ds_val)], ignore_index=True)

    # Ensure labels are numeric {0,1,2} and text column is named 'sentence'
    if "label" not in df.columns:
        raise ValueError("Dataset missing 'label' column.")
    if not set(df["label"].unique()).issubset({0, 1, 2}):
        raise ValueError(f"Unexpected numeric labels: {df['label'].unique()}")
    df["label"] = df["label"].astype(int)

    text_col = _find_text_column(df)
    if text_col != "sentence":
        df = df.rename(columns={text_col: "sentence"})

    # Keep only what's needed
    df = df[["sentence", "label"]].copy()
    return df

def stratified_split(data):
    """
    80/10/10 split with stratification on label, using SEED for reproducibility.
    """
    train_df, temp_df = train_test_split(
        data, test_size=0.2, stratify=data["label"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=SEED
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

def select_seed_data(df):
    """
    Sample ONLY from the training split.
    If USE_CLASS_BALANCED_SAMPLING, sample SAMPLES_PER_CLASS from each class (no replacement).
    Otherwise, sample a flat total of 105.
    """
    if USE_CLASS_BALANCED_SAMPLING:
        frames = []
        for label in sorted(df["label"].unique()):
            subset = df[df["label"] == label]
            if len(subset) < SAMPLES_PER_CLASS:
                raise ValueError(
                    f"Not enough examples for label {label}: have {len(subset)}, need {SAMPLES_PER_CLASS}"
                )
            chosen = subset.sample(n=SAMPLES_PER_CLASS, random_state=SEED, replace=False)
            frames.append(chosen)
        seed_df = pd.concat(frames).sample(frac=1.0, random_state=SEED).reset_index(drop=True)  # shuffle combined
    else:
        total = 105  # keep parity with your earlier setup
        if len(df) < total:
            raise ValueError(f"Not enough rows in training split to sample {total}.")
        seed_df = df.sample(n=total, random_state=SEED, replace=False).reset_index(drop=True)
    return seed_df

def convert_to_instruction_format(row):
    # Map twitter-financial-news-sentiment numeric labels to canonical names
    label_map = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
    return {
        "instruction": "Classify the sentiment of the financial sentence.",
        "input": row["sentence"],
        "output": label_map[row["label"]],
    }

def save_as_jsonl(df, path, seed_value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        # Write a header line with metadata (handy for provenance)
        f.write(json.dumps({"seed_used": seed_value, "source": "zeroshot/twitter-financial-news-sentiment"}) + "\n")
        for _, row in df.iterrows():
            f.write(json.dumps(convert_to_instruction_format(row), ensure_ascii=False) + "\n")
    print(f"Seed data (and seed value) saved to: {path}")

def main():
    set_seed(SEED)
    print(f"Using random seed: {SEED}")

    print("Loading full twitter-financial-news-sentiment dataset (train+validation)...")
    df = load_data()
    print(f"Combined size: {len(df)}")

    print("Stratified 80/10/10 split to avoid leakage...")
    train_df, _, _ = stratified_split(df)
    print(f"Training split size: {len(train_df)}")

    print("Selecting RANDOM seed data from the training split only...")
    seed_df = select_seed_data(train_df)
    print(f"Selected seed size: {len(seed_df)} (class-balanced={USE_CLASS_BALANCED_SAMPLING})")

    save_as_jsonl(seed_df, SAVE_PATH, SEED)

if __name__ == "__main__":
    main()
