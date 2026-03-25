import os
import pandas as pd
import numpy as np
import json
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Configuration
SAMPLES_PER_CLASS = 35          # random samples per class
USE_CLASS_BALANCED_SAMPLING = True
SEED = 24266 #24266 #26413 #36273
SAVE_PATH = "../outputs/seed_data_random.jsonl"   # fixed typo: random

def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)

def load_data():
    dataset = load_dataset(
        "takala/financial_phrasebank",
        "sentences_allagree",
        split="train",
        trust_remote_code=True
    )
    return pd.DataFrame(dataset)

def stratified_split(data):
    train_df, temp_df = train_test_split(
        data, test_size=0.2,
        stratify=data["label"],
        random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5,
        stratify=temp_df["label"],
        random_state=SEED
    )
    return train_df, val_df, test_df

def select_seed_data(df):
    """Pure random selection, reproducible via SEED."""
    if USE_CLASS_BALANCED_SAMPLING:
        frames = []
        for label in sorted(df["label"].unique()):
            subset = df[df["label"] == label]
            chosen = subset.sample(
                n=SAMPLES_PER_CLASS,
                random_state=SEED,
                replace=False
            )
            frames.append(chosen)
        seed_df = pd.concat(frames).reset_index(drop=True)
    else:
        total = 105  # same total as before
        seed_df = df.sample(n=total, random_state=SEED, replace=False).reset_index(drop=True)
    return seed_df

def convert_to_instruction_format(row):
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return {
        "instruction": "Classify the sentiment of the financial sentence.",
        "input": row["sentence"],
        "output": label_map[row["label"]]
    }

def save_as_jsonl(df, path, seed_value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"seed_used": seed_value}) + "\n")
        for _, row in df.iterrows():
            f.write(json.dumps(convert_to_instruction_format(row), ensure_ascii=False) + "\n")
    print(f"Seed data (and seed value) saved to: {path}")

def main():
    set_seed(SEED)
    print(f"Using random seed: {SEED}")

    print("Loading full dataset...")
    df = load_data()
    print(f"Dataset size: {len(df)}")

    print("Splitting data to avoid leakage...")
    train_df, _, _ = stratified_split(df)
    print(f"Training split size: {len(train_df)}")

    print("Selecting RANDOM seed data from training set only...")
    seed_df = select_seed_data(train_df)
    print(f"Selected seed size: {len(seed_df)}")

    save_as_jsonl(seed_df, SAVE_PATH, SEED)

if __name__ == "__main__":
    main()
