import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# === CONFIGURATION ===
SAMPLES_PER_CLASS = 35
USE_CLASS_BALANCED_CLUSTERING = True
SEED = 24266
FIGSIZE = (12, 6)
SAVE_DPI = 300
TITLE_FONTSIZE = 20
LEGEND_FONTSIZE = 16
SAVE_BBOX = 'tight'
SAVE_PAD  = 0.03  # smaller = thinner white border
LIM_PAD_FRAC = 0.06  # padding around min/max to avoid clipping markers

os.makedirs("../outputs", exist_ok=True)

def set_seed(seed=24266):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# Reduce figure border by default (still pass explicitly in savefig)
plt.rcParams.update({
    "savefig.bbox": SAVE_BBOX,
    "savefig.pad_inches": SAVE_PAD,
})

# === LABELS & COLORS ===
LABEL_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}
COLOR_ALL = {0: 'lightcoral', 1: 'lightblue', 2: 'lightgreen'}
COLOR_SEED = {0: '#E41A1C', 1: '#377EB8', 2: '#4DAF4A'}
COLOR_SEED_PC = {0: 'darkred', 1: 'darkblue', 2: 'darkgreen'}

COLOR_REAL = {0: 'skyblue', 1: 'lightgray', 2: 'lightgreen'}
COLOR_SYN  = {0: 'blue',     1: 'black',     2: 'green'}
COLOR_SEED_X = {0: 'red', 1: 'orange', 2: 'purple'}

# === HELPERS ===
def _new_fig():
    # layout='constrained' shrinks whitespace; use per-figure for consistency
    return plt.figure(figsize=FIGSIZE, layout='constrained')

def _prep_ax_equal(ax):
    ax.set_aspect('equal', adjustable='box')

def _expand_limits(vmin, vmax, frac=LIM_PAD_FRAC):
    span = vmax - vmin
    pad = span * frac if span > 0 else frac
    return vmin - pad, vmax + pad

# === FUNCTIONS ===
def load_data():
    dataset = load_dataset("takala/financial_phrasebank", "sentences_allagree",
                           split="train", trust_remote_code=True)
    return pd.DataFrame(dataset)

def generate_embeddings(sentences):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model.encode(sentences, show_progress_bar=True)

def cluster_embeddings(embeddings, num_clusters):
    model = KMeans(n_clusters=num_clusters, random_state=SEED)
    model.fit(embeddings)
    return model

def select_seed_data_with_distance(df):
    embeddings_all = generate_embeddings(df["sentence"].tolist())

    seed_indices, distances = [], []
    if USE_CLASS_BALANCED_CLUSTERING:
        for label in sorted(df['label'].unique()):
            cls_idx = np.where(df['label'].values == label)[0]
            k = min(SAMPLES_PER_CLASS, len(cls_idx))
            km = KMeans(n_clusters=k, random_state=SEED).fit(embeddings_all[cls_idx])
            closest, dists = pairwise_distances_argmin_min(km.cluster_centers_, embeddings_all[cls_idx])
            seed_indices.extend(cls_idx[closest].tolist())
            distances.extend(dists.tolist())
    else:
        num_classes = df['label'].nunique()
        total_seeds = min(SAMPLES_PER_CLASS * num_classes, len(df))
        km = KMeans(n_clusters=total_seeds, random_state=SEED).fit(embeddings_all)
        closest, dists = pairwise_distances_argmin_min(km.cluster_centers_, embeddings_all)
        seed_indices = closest.tolist()
        distances = dists.tolist()

    seed_df = df.iloc[seed_indices].reset_index(drop=True)
    return seed_df, distances, seed_indices, embeddings_all

def plot_distance_histogram(distances):
    _new_fig()
    plt.hist(distances, bins=25, color='skyblue', edgecolor='black')
    plt.title("Distances from Cluster Centers to Selected Seeds", fontsize=TITLE_FONTSIZE)
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Number of Samples")
    plt.grid(True)
    plt.savefig("../outputs/seed_distance_histogram.png", dpi=SAVE_DPI,
                bbox_inches=SAVE_BBOX, pad_inches=SAVE_PAD)
    plt.show()
    print("Saved: ../outputs/seed_distance_histogram.png")

def plot_tsne_projection(df, seed_indices, embeddings):
    reduced = TSNE(n_components=2, random_state=SEED).fit_transform(embeddings)

    # padded limits
    x_min, x_max = _expand_limits(reduced[:, 0].min(), reduced[:, 0].max())
    y_min, y_max = _expand_limits(reduced[:, 1].min(), reduced[:, 1].max())

    _new_fig()
    ax = plt.gca()
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    _prep_ax_equal(ax)

    for label in sorted(df['label'].unique()):
        idx = df[df['label'] == label].index
        plt.scatter(reduced[idx, 0], reduced[idx, 1],
                    c=COLOR_ALL[label], alpha=0.3, label=f"{LABEL_NAMES[label]} (All)")

    for label in sorted(df['label'].unique()):
        seed_sub = [i for i in seed_indices if df.iloc[i]['label'] == label]
        if seed_sub:
            plt.scatter(reduced[seed_sub, 0], reduced[seed_sub, 1],
                        c=COLOR_SEED[label], s=70, alpha=1.0, label=f"{LABEL_NAMES[label]} (Seed)")

    plt.title("t-SNE Projection of Embeddings with Seed Samples by Class", fontsize=TITLE_FONTSIZE)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.grid(True)
    plt.savefig("../outputs/tsne_projection_labeled.png", dpi=SAVE_DPI,
                bbox_inches=SAVE_BBOX, pad_inches=SAVE_PAD)
    plt.show()
    print("Saved: ../outputs/tsne_projection_labeled.png")

    # return padded limits for consistency downstream
    return reduced, (x_min, x_max), (y_min, y_max)

def plot_tsne_projection_per_class(df, seed_indices, reduced, xlim, ylim):
    for label in sorted(df['label'].unique()):
        _new_fig()
        ax = plt.gca()
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        _prep_ax_equal(ax)

        idx = df[df['label'] == label].index
        seed_sub = [i for i in seed_indices if df.iloc[i]['label'] == label]

        plt.scatter(reduced[idx, 0], reduced[idx, 1],
                    c=COLOR_ALL[label], alpha=0.3, label=f"{LABEL_NAMES[label]} (All)")
        if seed_sub:
            plt.scatter(reduced[seed_sub, 0], reduced[seed_sub, 1],
                        c=COLOR_SEED_PC[label], label=f"{LABEL_NAMES[label]} (Seed)")

        plt.title(f"t-SNE Projection : {LABEL_NAMES[label]}", fontsize=TITLE_FONTSIZE)
        plt.legend(fontsize=LEGEND_FONTSIZE)
        plt.grid(True)
        fname = f"../outputs/tsne_projection_{LABEL_NAMES[label].lower()}_filtered.png"
        plt.savefig(fname, dpi=SAVE_DPI, bbox_inches=SAVE_BBOX, pad_inches=SAVE_PAD)
        plt.show()
        print(f"Saved: {fname}")

def plot_tsne_combined_real_synthetic(real_df, synthetic_df):
    all_sentences = real_df["sentence"].tolist() + synthetic_df["sentence"].tolist()
    all_embeddings = generate_embeddings(all_sentences)
    reduced = TSNE(n_components=2, random_state=SEED).fit_transform(all_embeddings)

    real_len = len(real_df)

    # padded limits
    x_min, x_max = _expand_limits(reduced[:, 0].min(), reduced[:, 0].max())
    y_min, y_max = _expand_limits(reduced[:, 1].min(), reduced[:, 1].max())

    _new_fig()
    ax = plt.gca()
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    _prep_ax_equal(ax)

    plt.scatter(reduced[:real_len, 0], reduced[:real_len, 1],
                c='skyblue', alpha=0.4, label="Real Data")
    plt.scatter(reduced[real_len:, 0], reduced[real_len:, 1],
                c='orange', alpha=0.6, label="Synthetic Data")

    plt.title("t-SNE Projection: Real vs. Synthetic Data", fontsize=TITLE_FONTSIZE)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.grid(True)
    plt.savefig("../outputs/tsne_projection_real_vs_synthetic.png", dpi=SAVE_DPI,
                bbox_inches=SAVE_BBOX, pad_inches=SAVE_PAD)
    plt.show()
    print("Saved: ../outputs/tsne_projection_real_vs_synthetic.png")

def plot_tsne_per_class_real_synthetic(real_df, synthetic_df):
    for label in sorted(real_df['label'].unique()):
        real_class = real_df[real_df["label"] == label].reset_index(drop=True)
        synthetic_class = synthetic_df[synthetic_df["label"] == label].reset_index(drop=True)

        all_sentences = real_class["sentence"].tolist() + synthetic_class["sentence"].tolist()
        embeddings = generate_embeddings(all_sentences)
        reduced = TSNE(n_components=2, random_state=SEED).fit_transform(embeddings)

        real_len = len(real_class)

        # padded limits
        x_min, x_max = _expand_limits(reduced[:, 0].min(), reduced[:, 0].max())
        y_min, y_max = _expand_limits(reduced[:, 1].min(), reduced[:, 1].max())

        _new_fig()
        ax = plt.gca()
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        _prep_ax_equal(ax)

        plt.scatter(reduced[:real_len, 0], reduced[:real_len, 1],
                    c=COLOR_REAL[label], alpha=0.4, label=f"{LABEL_NAMES[label]} (Real)")
        plt.scatter(reduced[real_len:, 0], reduced[real_len:, 1],
                    c=COLOR_SYN[label], alpha=0.7, label=f"{LABEL_NAMES[label]} (Synthetic)")

        plt.title(f"t-SNE Projection: {LABEL_NAMES[label]} – Real vs Synthetic", fontsize=TITLE_FONTSIZE)
        plt.legend(fontsize=LEGEND_FONTSIZE)
        plt.grid(True)
        fname = f"../outputs/tsne_projection_real_vs_synthetic_{LABEL_NAMES[label].lower()}.png"
        plt.savefig(fname, dpi=SAVE_DPI, bbox_inches=SAVE_BBOX, pad_inches=SAVE_PAD)
        plt.show()
        print(f"Saved: {fname}")

def plot_tsne_per_class_real_synthetic_with_seed_overlay(real_df, synthetic_df, seed_df):
    for label in sorted(real_df['label'].unique()):
        real_class = real_df[real_df["label"] == label].reset_index(drop=True)
        synthetic_class = synthetic_df[synthetic_df["label"] == label].reset_index(drop=True)
        seed_class = seed_df[seed_df["label"] == label].reset_index(drop=True)

        all_sentences = real_class["sentence"].tolist() + synthetic_class["sentence"].tolist()
        all_embeddings = generate_embeddings(all_sentences)
        reduced = TSNE(n_components=2, random_state=SEED).fit_transform(all_embeddings)

        real_len = len(real_class)
        synth_len = len(synthetic_class)

        # sanity check
        assert reduced.shape[0] == real_len + synth_len, "t-SNE rows != real_len + synth_len"

        # padded limits
        x_min, x_max = _expand_limits(reduced[:, 0].min(), reduced[:, 0].max())
        y_min, y_max = _expand_limits(reduced[:, 1].min(), reduced[:, 1].max())

        _new_fig()
        ax = plt.gca()
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        _prep_ax_equal(ax)

        plt.scatter(reduced[:real_len, 0], reduced[:real_len, 1],
                    c=COLOR_REAL[label], alpha=0.3, label=f"{LABEL_NAMES[label]} (Real)")
        plt.scatter(reduced[real_len:, 0], reduced[real_len:, 1],
                    c=COLOR_SYN[label], alpha=0.8, label=f"{LABEL_NAMES[label]} (Synthetic)")

        if not seed_class.empty:
            seed_mask = real_class["sentence"].isin(seed_class["sentence"])
            seed_idx = np.flatnonzero(seed_mask.values)
            if seed_idx.size > 0:
                plt.scatter(reduced[seed_idx, 0], reduced[seed_idx, 1],
                            c=COLOR_SEED_X[label], marker='x', s=90, label=f"{LABEL_NAMES[label]} (Seed)")

        plt.title(f"t-SNE Projection: {LABEL_NAMES[label]} – Real, Synthetic, Seed", fontsize=TITLE_FONTSIZE)
        plt.legend(fontsize=LEGEND_FONTSIZE)
        plt.grid(True)
        fname = f"../outputs/tsne_projection_real_vs_synthetic_{LABEL_NAMES[label].lower()}_with_seed_overlay.png"
        plt.savefig(fname, dpi=SAVE_DPI, bbox_inches=SAVE_BBOX, pad_inches=SAVE_PAD)
        plt.show()
        print(f"Saved: {fname}")

# === MAIN ===
if __name__ == "__main__":
    print("Loading data...")
    data = load_data()

    print("Selecting seed data...")
    seed_df, distances, seed_indices, full_embeddings = select_seed_data_with_distance(data)

    print("Generating visualizations...")
    plot_distance_histogram(distances)
    reduced, xlim, ylim = plot_tsne_projection(data, seed_indices, full_embeddings)
    plot_tsne_projection_per_class(data, seed_indices, reduced, xlim, ylim)

    print("Loading synthetic data...")
    with open("../outputs/synthetic_data_from_Seed.jsonl", "r", encoding="utf-8") as f:
        synthetic_lines = [json.loads(line.strip()) for line in f]

    synthetic_df = pd.DataFrame([
        {"sentence": d["input"].strip(),
         "label": {"Negative": 0, "Neutral": 1, "Positive": 2}[d["output"].strip()]}
        for d in synthetic_lines if "input" in d and "output" in d
    ])

    print("Generating real vs. synthetic t-SNE plot (all classes)...")
    plot_tsne_combined_real_synthetic(data, synthetic_df)

    print("Generating per-class real vs. synthetic t-SNE plots...")
    plot_tsne_per_class_real_synthetic(data, synthetic_df)

    print("Generating per-class real vs. synthetic t-SNE plots WITH seed...")
    plot_tsne_per_class_real_synthetic_with_seed_overlay(data, synthetic_df, seed_df)

    print("Done.")
