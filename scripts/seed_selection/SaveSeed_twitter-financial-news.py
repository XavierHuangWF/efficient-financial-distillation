import os
import pandas as pd
import numpy as np
import json
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.model_selection import train_test_split

# Configuration
SAMPLES_PER_CLASS = 35
USE_CLASS_BALANCED_CLUSTERING = True
SEED = 24266
SAVE_PATH = "../../outputs/seed_data.jsonl"

def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)

# Canonical order: 0=Bearish, 1=Bullish, 2=Neutral
CLASS_NAMES = {0: "Bearish", 1: "Bullish", 2: "Neutral"}

# === Load the SAME POOL as the second script ===
def load_data_same_pool():
    """
    Matches the second script's DataLoad():
    - Load 'train' and 'validation' splits
    - Concatenate them
    - Rename 'text' -> 'sentence'
    - Ensure labels are ints 0/1/2
    """
    ds_train = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
    ds_val = load_dataset("zeroshot/twitter-financial-news-sentiment", split="validation")
    df = pd.concat([pd.DataFrame(ds_train), pd.DataFrame(ds_val)], ignore_index=True)

    # Standardize column names to match downstream code
    if "sentence" not in df.columns:
        if "text" in df.columns:
            df = df.rename(columns={"text": "sentence"})
        else:
            # Fallbacks (rare for this dataset, but safe to have)
            for alt in ["tweet", "content", "document", "message", "headline"]:
                if alt in df.columns:
                    df = df.rename(columns={alt: "sentence"})
                    break

    if "sentence" not in df.columns:
        raise ValueError("Could not find a text column ('text' or compatible) in dataset.")

    # Ensure label is int (0=Bearish, 1=Bullish, 2=Neutral)
    if "label" not in df.columns:
        # Very unlikely for this dataset, but guard anyway
        for alt in ["sentiment", "target", "y", "labels", "label_id"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "label"})
                break
    if "label" not in df.columns:
        raise ValueError("Could not find a label column in dataset.")

    df["label"] = df["label"].astype(int)

    return df[["sentence", "label"]]

def stratified_split(data):
    train_df, temp_df = train_test_split(
        data, test_size=0.2, stratify=data["label"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=SEED
    )
    return train_df, val_df, test_df

def generate_embeddings(sentences):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model.encode(sentences, show_progress_bar=True)

def cluster_embeddings(embeddings, num_clusters):
    model = KMeans(n_clusters=num_clusters, random_state=SEED)
    model.fit(embeddings)
    return model

def select_seed_data(df):
    if USE_CLASS_BALANCED_CLUSTERING:
        seed_frames = []
        for label in sorted(df['label'].unique()):
            subset = df[df['label'] == label].reset_index(drop=True)
            embeddings = generate_embeddings(subset["sentence"].tolist())
            cluster_model = cluster_embeddings(embeddings, num_clusters=SAMPLES_PER_CLASS)
            closest, _ = pairwise_distances_argmin_min(cluster_model.cluster_centers_, embeddings)
            selected = subset.iloc[closest]
            seed_frames.append(selected)
        seed_df = pd.concat(seed_frames).reset_index(drop=True)
    else:
        # Non-balanced option: total 105 seeds
        embeddings = generate_embeddings(df["sentence"].tolist())
        cluster_model = cluster_embeddings(embeddings, num_clusters=105)
        closest, _ = pairwise_distances_argmin_min(cluster_model.cluster_centers_, embeddings)
        seed_df = df.iloc[closest].reset_index(drop=True)
    return seed_df

def convert_to_instruction_format(row):
    return {
        "instruction": "Classify the sentiment of the financial text as Bearish, Bullish, or Neutral.",
        "input": row["sentence"],
        "output": CLASS_NAMES[row["label"]]
    }

def save_as_jsonl(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            json_line = convert_to_instruction_format(row)
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
    print(f"Seed data saved to: {path}")

def main():
    set_seed()
    print("Loading combined dataset (train + validation) to MATCH the second script...")
    df = load_data_same_pool()
    print(f"Combined dataset size (train+val): {len(df)}")

    print("Stratified split (same seed) to avoid leakage...")
    train_df, _, _ = stratified_split(df)
    print(f"Training split size: {len(train_df)}")

    print("Selecting seed data from training set only (class-balanced, 35 per class)...")
    seed_df = select_seed_data(train_df)
    print(f"Selected seed size: {len(seed_df)}")

    save_as_jsonl(seed_df, SAVE_PATH)

if __name__ == "__main__":
    main()
