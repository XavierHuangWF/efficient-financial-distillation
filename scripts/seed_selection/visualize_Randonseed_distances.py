import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# === CONFIGURATION ===
SAMPLES_PER_CLASS = 35
USE_CLASS_BALANCED_CLUSTERING = True   # (unused now; we load seeds from file)
SEED = 24266
FIGSIZE = (12, 6)
SAVE_DPI = 300
TITLE_FONTSIZE = 20
LEGEND_FONTSIZE = 16
SAVE_BBOX = 'tight'
SAVE_PAD  = 0.03  # smaller = thinner white border
LIM_PAD_FRAC = 0.06  # padding around min/max to avoid clipping markers

# Preferred local paths; fallback to /mnt/data if present
SEED_JSONL_CANDIDATES = [
    "../outputs/seed_data.jsonl"
]
SYN_JSONL_CANDIDATES = [
    "../outputs/synthetic_data_from_Seed.jsonl"
]

os.makedirs("../outputs", exist_ok=True)

def set_seed(seed=24266):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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

def _resolve_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of the candidate paths exist: {paths}")

# === IO FUNCTIONS (JSONL) ===
def load_seed_jsonl(path):
    """
    Loads seed JSONL written by your generator.
    First line may be {"seed_used": ...}; we skip any line missing input/output.
    Returns DataFrame with columns: sentence, label (0/1/2).
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "input" in obj and "output" in obj:
                sent = str(obj["input"]).strip()
                out = str(obj["output"]).strip()
                label = {"Negative": 0, "Neutral": 1, "Positive": 2}.get(out, None)
                if label is not None:
                    rows.append({"sentence": sent, "label": label})
            # else: skip (likely {"seed_used": ...})
    if not rows:
        raise ValueError(f"No valid seed rows found in {path}")
    return pd.DataFrame(rows)

def load_synth_jsonl(path):
    """
    Loads synthetic JSONL (input/output keys).
    Returns DataFrame with columns: sentence, label (0/1/2).
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "input" in obj and "output" in obj:
                sent = str(obj["input"]).strip()
                out = str(obj["output"]).strip()
                label = {"Negative": 0, "Neutral": 1, "Positive": 2}.get(out, None)
                if label is not None:
                    rows.append({"sentence": sent, "label": label})
    if not rows:
        raise ValueError(f"No valid synthetic rows found in {path}")
    return pd.DataFrame(rows)

# === DATA FUNCTIONS ===
def load_data():
    dataset = load_dataset("takala/financial_phrasebank", "sentences_allagree",
                           split="train", trust_remote_code=True)
    return pd.DataFrame(dataset)

def generate_embeddings(sentences):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model.encode(sentences, show_progress_bar=True)

# === PLOTTING ===
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

    # Overlay seeds
    for label in sorted(df['label'].unique()):
        seed_sub = [i for i in seed_indices if df.iloc[i]['label'] == label]
        if seed_sub:
            plt.scatter(reduced[seed_sub, 0], reduced[seed_sub, 1],
                        c=COLOR_SEED[label], s=70, alpha=1.0, label=f"{LABEL_NAMES[label]} (Seed)")

    plt.title("t-SNE Projection of Embeddings with Random Seed Samples by Class", fontsize=TITLE_FONTSIZE)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.grid(True)
    plt.savefig("../outputs/tsne_projection_labeled_random_seed.png", dpi=SAVE_DPI,
                bbox_inches=SAVE_BBOX, pad_inches=SAVE_PAD)
    plt.show()
    print("Saved: ../outputs/tsne_projection_labeled_random_seed.png")

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

        plt.title(f"t-SNE Projection : {LABEL_NAMES[label]} (Random Seeds)", fontsize=TITLE_FONTSIZE)
        plt.legend(fontsize=LEGEND_FONTSIZE)
        plt.grid(True)
        fname = f"../outputs/tsne_projection_{LABEL_NAMES[label].lower()}_filtered_random_seed.png"
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

        # Overlay seed (found by sentence match inside the real_class slice)
        if not seed_class.empty:
            seed_mask = real_class["sentence"].isin(seed_class["sentence"])
            seed_idx = np.flatnonzero(seed_mask.values)
            if seed_idx.size > 0:
                plt.scatter(reduced[seed_idx, 0], reduced[seed_idx, 1],
                            c=COLOR_SEED_X[label], marker='x', s=90, label=f"{LABEL_NAMES[label]} (Seed)")

        plt.title(f"t-SNE Projection: {LABEL_NAMES[label]} – Real, Synthetic, Seed (Random)", fontsize=TITLE_FONTSIZE)
        plt.legend(fontsize=LEGEND_FONTSIZE)
        plt.grid(True)
        fname = f"../outputs/tsne_projection_real_vs_synthetic_{LABEL_NAMES[label].lower()}_with_seed_overlay_random.png"
        plt.savefig(fname, dpi=SAVE_DPI, bbox_inches=SAVE_BBOX, pad_inches=SAVE_PAD)
        plt.show()
        print(f"Saved: {fname}")

# === MAIN ===
if __name__ == "__main__":
    print("Loading real data...")
    data = load_data()

    print("Loading random seed JSONL...")
    seed_path = _resolve_first_existing(SEED_JSONL_CANDIDATES)
    seed_df = load_seed_jsonl(seed_path)
    print(f"Loaded seeds: {len(seed_df)} from {seed_path}")

    print("Loading synthetic JSONL...")
    syn_path = _resolve_first_existing(SYN_JSONL_CANDIDATES)
    synthetic_df = load_synth_jsonl(syn_path)
    print(f"Loaded synthetic: {len(synthetic_df)} from {syn_path}")

    print("Embedding all real sentences (one pass)...")
    full_embeddings = generate_embeddings(data["sentence"].tolist())

    # Find indices of seed sentences within the real data
    # (If duplicates exist, this includes all matching rows.)
    seed_mask_global = data["sentence"].isin(seed_df["sentence"])
    seed_indices = data.index[seed_mask_global].tolist()
    print(f"Matched {len(seed_indices)} seed indices in real data.")

    print("Generating visualizations...")
    reduced, xlim, ylim = plot_tsne_projection(data, seed_indices, full_embeddings)
    plot_tsne_projection_per_class(data, seed_indices, reduced, xlim, ylim)

    print("Generating real vs. synthetic t-SNE plot (all classes)...")
    plot_tsne_combined_real_synthetic(data, synthetic_df)

    print("Generating per-class real vs. synthetic t-SNE plots...")
    plot_tsne_per_class_real_synthetic(data, synthetic_df)

    print("Generating per-class real vs. synthetic t-SNE plots WITH random seed overlay...")
    plot_tsne_per_class_real_synthetic_with_seed_overlay(data, synthetic_df, seed_df)

    print("Done.")
