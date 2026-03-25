import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# ======================
# CONFIGURATION
# ======================
DATA_SOURCE = "twitter"  # "twitter" or "phrasebank"
SEED = 24266
FIGSIZE = (12, 6)
SAVE_DPI = 300
TITLE_FONTSIZE = 20
LEGEND_FONTSIZE = 16
SAVE_BBOX = 'tight'
SAVE_PAD  = 0.03
LIM_PAD_FRAC = 0.06

# Global toggles
SHOW_TITLES = True    # set True if you want titles
SHOW_LEGENDS = True    # legends ON

# File paths (Twitter defaults)
SEED_JSONL_CLUSTERED_CANDIDATES = [
    "../outputs/seed_data.jsonl"
]
SEED_JSONL_RANDOM_CANDIDATES = [
    "../outputs/1seed_data_twitter_random.jsonl"
]
SYN_JSONL_CLUSTERED_CANDIDATES = [
    "../outputs/synthetic_data_from_Seed.jsonl"
]
SYN_JSONL_RANDOM_CANDIDATES = [
    "../outputs/1synthetic_data_from_Seed_random.jsonl"
]

os.makedirs("../outputs", exist_ok=True)

def set_seed(seed=SEED):
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

plt.rcParams.update({
    "savefig.bbox": SAVE_BBOX,
    "savefig.pad_inches": SAVE_PAD,
})

# ======================
# COLORS / LABELS
# ======================
LABEL_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}
COLOR_ALL = {0: 'lightcoral', 1: 'lightblue', 2: 'lightgreen'}
COLOR_SEED = {0: '#E41A1C', 1: '#377EB8', 2: '#4DAF4A'}
COLOR_SEED_PC = {0: 'darkred', 1: 'darkblue', 2: 'darkgreen'}

COLOR_REAL = {0: 'skyblue', 1: 'lightgray', 2: 'lightgreen'}
COLOR_SYN  = {0: 'blue',     1: 'black',     2: 'green'}

COLOR_SEED_RANDOM  = {0: '#D62728', 1: '#FF7F0E', 2: '#9467BD'}  # x markers
COLOR_SEED_CLUSTER = {0: '#1F77B4', 1: '#2CA02C', 2: '#17BECF'}  # squares

# ======================
# HELPERS
# ======================
def _new_fig():
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

def _maybe_title(text):
    if SHOW_TITLES and text:
        plt.title(text, fontsize=TITLE_FONTSIZE)

def _apply_legend(ax, ncol=1):
    """Show legend if enabled; dedupe entries & keep order."""
    if not SHOW_LEGENDS:
        return
    handles, labels = ax.get_legend_handles_labels()
    if not labels:
        return
    seen = {}
    dedup_h, dedup_l = [], []
    for h, l in zip(handles, labels):
        if l and l not in seen:
            seen[l] = True
            dedup_h.append(h)
            dedup_l.append(l)
    ax.legend(dedup_h, dedup_l, fontsize=LEGEND_FONTSIZE, ncol=ncol)

def stratified_split_train_val_test(df, seed=SEED):
    """80/10/10 stratified split; returns (train_df, val_df, test_df)."""
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=seed
    )
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))

def _warn_filtered(seeds_df, idx_list, name):
    matched = len(idx_list)
    total = len(seeds_df)
    if matched < total:
        print(f"[WARN] {name}: {matched}/{total} seeds are in TRAIN "
              f"({total - matched} dropped because they’re not in the train split).")

# ======================
# JSONL LOADING
# ======================
def _label_from_output(out: str):
    # Accept multiple vocabularies
    m = {
        # standard
        "negative": 0, "neg": 0,
        "neutral": 1,  "neu": 1,
        "positive": 2, "pos": 2,
        # capitalization variants
        "Negative": 0, "Neutral": 1, "Positive": 2,
        "Neg": 0, "Neu": 1, "Pos": 2,
        # twitter-specific
        "Bearish": 0, "bearish": 0,
        "Bullish": 2, "bullish": 2,
    }
    if out in m:
        return m[out]
    try:
        val = int(out)
        return {-1: 0, 0: 1, 1: 2, 2: 2}.get(val, None)
    except Exception:
        return None

def load_seed_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "input" in obj and "output" in obj:
                sent = str(obj["input"]).strip()
                label = _label_from_output(str(obj["output"]).strip())
                if label is not None:
                    rows.append({"sentence": sent, "label": label})
    if not rows:
        raise ValueError(f"No valid seed rows found in {path}")
    return pd.DataFrame(rows)

def load_synth_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "input" in obj and "output" in obj:
                sent = str(obj["input"]).strip()
                label = _label_from_output(str(obj["output"]).strip())
                if label is not None:
                    rows.append({"sentence": sent, "label": label})
    if not rows:
        raise ValueError(f"No valid synthetic rows found in {path}")
    return pd.DataFrame(rows)

# ======================
# DATA LOADING (same POOL as your seed generator)
# ======================
def _normalize_labels(df, text_col, label_col):
    """Map labels to 0:Neg, 1:Neu, 2:Pos; return standardized DataFrame."""
    series = df[label_col]
    if series.dtype == object:
        mapped = series.astype(str).str.lower().map(
            {"negative": 0, "neg": 0, "neutral": 1, "neu": 1, "positive": 2, "pos": 2,
             "bearish": 0, "bullish": 2}
        )
    else:
        mapped = series
        try:
            mapped = mapped.astype(int)
        except Exception:
            pass

    if np.issubdtype(mapped.dtype, np.integer):
        uniq = set(pd.unique(mapped))
        if uniq <= {-1, 0, 1}:
            mapped = mapped.map({-1: 0, 0: 1, 1: 2})
        elif not (uniq <= {0, 1, 2}):
            mapped = mapped.replace({-1: 0, 0: 1, 1: 2})

    out = pd.DataFrame({"sentence": df[text_col].astype(str).str.strip(),
                        "label": mapped.astype(int)})
    out = out[out["label"].isin([0, 1, 2])].reset_index(drop=True)
    if out.empty:
        raise ValueError("Could not normalize labels to {0,1,2}.")
    return out

def load_data_phrasebank():
    dset = load_dataset("takala/financial_phrasebank", "sentences_allagree",
                        split="train", trust_remote_code=True)
    df = pd.DataFrame(dset)
    text_col = "sentence" if "sentence" in df.columns else ("text" if "text" in df.columns else df.columns[0])
    label_col = "label" if "label" in df.columns else "sentiment"
    return _normalize_labels(df, text_col, label_col)

def load_data_twitter():
    """
    EXACTLY match the seed-generator pool:
    concatenate HF train + validation from zeroshot/twitter-financial-news-sentiment,
    then we will do our own stratified split (80/10/10).
    """
    ds_train = load_dataset("zeroshot/twitter-financial-news-sentiment",
                            split="train", trust_remote_code=True)
    ds_val   = load_dataset("zeroshot/twitter-financial-news-sentiment",
                            split="validation", trust_remote_code=True)
    df = pd.concat([pd.DataFrame(ds_train), pd.DataFrame(ds_val)], ignore_index=True)

    # choose sensible text/label columns
    text_cols = [c for c in ["text", "tweet", "sentence", "content"] if c in df.columns]
    label_cols = [c for c in ["label", "sentiment", "target"] if c in df.columns]
    if not text_cols or not label_cols:
        raise ValueError("Could not find text/label columns in twitter dataset.")
    return _normalize_labels(df, text_cols[0], label_cols[0])

def load_data():
    if DATA_SOURCE.lower().startswith("twit"):
        return load_data_twitter()
    return load_data_phrasebank()

# ======================
# EMBEDDINGS
# ======================
def generate_embeddings(sentences):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model.encode(sentences, show_progress_bar=True)

# ======================
# PLOTTING
# ======================
def plot_tsne_projection(df, seed_indices, embeddings):
    reduced = TSNE(n_components=2, random_state=SEED).fit_transform(embeddings)
    x_min, x_max = _expand_limits(reduced[:, 0].min(), reduced[:, 0].max())
    y_min, y_max = _expand_limits(reduced[:, 1].min(), reduced[:, 1].max())

    _new_fig()
    ax = plt.gca()
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    _prep_ax_equal(ax)

    for label in sorted(df['label'].unique()):
        idx = df[df['label'] == label].index
        plt.scatter(reduced[idx, 0], reduced[idx, 1],
                    c=COLOR_ALL[label], alpha=0.3,
                    label=f"{LABEL_NAMES[label]} (All)")

    for label in sorted(df['label'].unique()):
        seed_sub = [i for i in seed_indices if df.iloc[i]['label'] == label]
        if seed_sub:
            plt.scatter(reduced[seed_sub, 0], reduced[seed_sub, 1],
                        c=COLOR_SEED[label], s=70, alpha=1.0,
                        label=f"{LABEL_NAMES[label]} (Seed)")

    _maybe_title("t-SNE Projection of Embeddings with Seed Samples by Class")
    _apply_legend(ax)
    plt.grid(True)
    plt.savefig("../outputs/tsne_projection_labeled.png", dpi=SAVE_DPI,
                bbox_inches=SAVE_BBOX, pad_inches=SAVE_PAD)
    plt.close()
    return reduced, (x_min, x_max), (y_min, y_max)

def plot_tsne_projection_per_class(df, seed_indices, reduced, xlim, ylim, file_tag=""):
    for label in sorted(df['label'].unique()):
        _new_fig()
        ax = plt.gca()
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        _prep_ax_equal(ax)

        idx = df[df['label'] == label].index
        seed_sub = [i for i in seed_indices if df.iloc[i]['label'] == label]

        plt.scatter(reduced[idx, 0], reduced[idx, 1],
                    c=COLOR_ALL[label], alpha=0.3,
                    label=f"{LABEL_NAMES[label]} (All)")
        if seed_sub:
            plt.scatter(reduced[seed_sub, 0], reduced[seed_sub, 1],
                        c=COLOR_SEED_PC[label],
                        label=f"{LABEL_NAMES[label]} (Seed)")

        _maybe_title(f"t-SNE Projection : {LABEL_NAMES[label]}")
        _apply_legend(ax)
        plt.grid(True)
        fname = f"../outputs/tsne_projection_{LABEL_NAMES[label].lower()}_filtered{file_tag}.png"
        plt.savefig(fname, dpi=SAVE_DPI, bbox_inches=SAVE_BBOX, pad_inches=SAVE_PAD)
        plt.close()

def plot_tsne_projection_with_given_reduction(df, seed_indices, reduced, xlim, ylim,
                                              title_suffix="", outfile_suffix=""):
    _new_fig()
    ax = plt.gca()
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    _prep_ax_equal(ax)

    for label in sorted(df['label'].unique()):
        idx = df[df['label'] == label].index
        plt.scatter(reduced[idx, 0], reduced[idx, 1],
                    c=COLOR_ALL[label], alpha=0.3,
                    label=f"{LABEL_NAMES[label]} (All)")

    for label in sorted(df['label'].unique()):
        seed_sub = [i for i in seed_indices if df.iloc[i]['label'] == label]
        if seed_sub:
            plt.scatter(reduced[seed_sub, 0], reduced[seed_sub, 1],
                        c=COLOR_SEED[label], s=70, alpha=1.0,
                        label=f"{LABEL_NAMES[label]} (Seed)")

    _maybe_title(f"t-SNE Projection of Embeddings with Seed Samples {title_suffix}".strip())
    _apply_legend(ax)
    plt.grid(True)
    out = f"../outputs/tsne_projection_labeled{outfile_suffix}.png"
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches=SAVE_BBOX, pad_inches=SAVE_PAD)
    plt.close()

def plot_tsne_projection_per_class_with_given_reduction(df, seed_indices, reduced, xlim, ylim,
                                                        title_tag, file_tag):
    for label in sorted(df['label'].unique()):
        _new_fig()
        ax = plt.gca()
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        _prep_ax_equal(ax)

        idx = df[df['label'] == label].index
        seed_sub = [i for i in seed_indices if df.iloc[i]['label'] == label]

        plt.scatter(reduced[idx, 0], reduced[idx, 1],
                    c=COLOR_ALL[label], alpha=0.3,
                    label=f"{LABEL_NAMES[label]} (All)")
        if seed_sub:
            plt.scatter(reduced[seed_sub, 0], reduced[seed_sub, 1],
                        c=COLOR_SEED_PC[label],
                        label=f"{LABEL_NAMES[label]} (Seed)")

        _maybe_title(f"t-SNE Projection : {LABEL_NAMES[label]} ({title_tag})")
        _apply_legend(ax)
        plt.grid(True)
        fname = f"../outputs/tsne_projection_{LABEL_NAMES[label].lower()}_filtered{file_tag}.png"
        plt.savefig(fname, dpi=SAVE_DPI, bbox_inches=SAVE_BBOX, pad_inches=SAVE_PAD)
        plt.close()

def plot_tsne_overlay_two_seed_sets(df, reduced, xlim, ylim, seed_idx_random, seed_idx_clustered):
    _new_fig()
    ax = plt.gca()
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    _prep_ax_equal(ax)

    # Simple legend (one handle per type)
    plt.scatter([], [], c='gray', alpha=0.25, label="All")
    plt.scatter([], [], c='k', marker='x', s=90, label="Random")
    plt.scatter([], [], c='k', marker='s', s=55, label="Clustered")

    # background
    for label in sorted(df['label'].unique()):
        idx = df[df['label'] == label].index
        plt.scatter(reduced[idx, 0], reduced[idx, 1],
                    c=COLOR_ALL[label], alpha=0.25)

    # random seeds
    for label in sorted(df['label'].unique()):
        sub = [i for i in seed_idx_random if df.iloc[i]['label'] == label]
        if sub:
            plt.scatter(reduced[sub, 0], reduced[sub, 1],
                        c=COLOR_SEED_RANDOM[label], marker='x', s=90)

    # clustered seeds
    for label in sorted(df['label'].unique()):
        sub = [i for i in seed_idx_clustered if df.iloc[i]['label'] == label]
        if sub:
            plt.scatter(reduced[sub, 0], reduced[sub, 1],
                        c=COLOR_SEED_CLUSTER[label], marker='s', s=55)

    _maybe_title("t-SNE (fixed) — Random vs Clustered seed overlays")
    _apply_legend(ax, ncol=2)
    plt.grid(True)
    out = "../outputs/tsne_overlay_random_vs_clustered.png"
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches=SAVE_BBOX, pad_inches=SAVE_PAD)
    plt.close()

def plot_tsne_combined_real_synthetic(real_df, synthetic_df, outname="tsne_projection_real_vs_synthetic.png"):
    all_sentences = real_df["sentence"].tolist() + synthetic_df["sentence"].tolist()
    all_embeddings = generate_embeddings(all_sentences)
    reduced = TSNE(n_components=2, random_state=SEED).fit_transform(all_embeddings)

    real_len = len(real_df)
    x_min, x_max = _expand_limits(reduced[:, 0].min(), reduced[:, 0].max())
    y_min, y_max = _expand_limits(reduced[:, 1].min(), reduced[:, 1].max())

    _new_fig()
    ax = plt.gca()
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    _prep_ax_equal(ax)

    plt.scatter(reduced[:real_len, 0], reduced[:real_len, 1],
                c='skyblue', alpha=0.4, label="Real")
    plt.scatter(reduced[real_len:, 0], reduced[real_len:, 1],   # <-- correct slice
                c='orange', alpha=0.6, label="Synthetic")

    _maybe_title("t-SNE Projection: Real vs. Synthetic Data")
    _apply_legend(ax)
    plt.grid(True)
    plt.savefig(f"../outputs/{outname}", dpi=SAVE_DPI,
                bbox_inches=SAVE_BBOX, pad_inches=SAVE_PAD)
    plt.close()

def plot_tsne_per_class_real_synthetic(real_df, synthetic_df, file_tag=""):
    for label in sorted(real_df['label'].unique()):
        real_class = real_df[real_df["label"] == label].reset_index(drop=True)
        synthetic_class = synthetic_df[synthetic_df["label"] == label].reset_index(drop=True)

        all_sentences = real_class["sentence"].tolist() + synthetic_class["sentence"].tolist()
        embeddings = generate_embeddings(all_sentences)
        reduced = TSNE(n_components=2, random_state=SEED).fit_transform(embeddings)

        real_len = len(real_class)
        x_min, x_max = _expand_limits(reduced[:, 0].min(), reduced[:, 0].max())
        y_min, y_max = _expand_limits(reduced[:, 1].min(), reduced[:, 1].max())

        _new_fig()
        ax = plt.gca()
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        _prep_ax_equal(ax)

        plt.scatter(reduced[:real_len, 0], reduced[:real_len, 1],
                    c=COLOR_REAL[label], alpha=0.4, label="Real")
        plt.scatter(reduced[real_len:, 0], reduced[real_len:, 1],   # <-- correct slice
                    c=COLOR_SYN[label], alpha=0.7, label="Synthetic")

        _maybe_title(f"t-SNE Projection: {LABEL_NAMES[label]} – Real vs Synthetic")
        _apply_legend(ax)
        plt.grid(True)
        fname = f"../outputs/tsne_projection_real_vs_synthetic_{LABEL_NAMES[label].lower()}{file_tag}.png"
        plt.savefig(fname, dpi=SAVE_DPI, bbox_inches=SAVE_BBOX, pad_inches=SAVE_PAD)
        plt.close()

# ======================
# MAIN
# ======================
if __name__ == "__main__":
    print(f"Loading real data from: {DATA_SOURCE}")
    data_pool = load_data()  # Twitter: train+validation concatenated (same pool as seed gen)
    train_df, val_df, test_df = stratified_split_train_val_test(data_pool, seed=SEED)
    data = train_df
    print(f"Using TRAIN split only for t-SNE plots: {len(data)} rows")

    print("Loading seed JSONLs (clustered & random)...")
    seed_cluster_path = _resolve_first_existing(SEED_JSONL_CLUSTERED_CANDIDATES)
    seed_random_path  = _resolve_first_existing(SEED_JSONL_RANDOM_CANDIDATES)
    seed_cluster_df = load_seed_jsonl(seed_cluster_path)
    seed_random_df  = load_seed_jsonl(seed_random_path)

    print("Loading synthetic JSONLs...")
    syn_cluster_path = _resolve_first_existing(SYN_JSONL_CLUSTERED_CANDIDATES)
    syn_random_path  = _resolve_first_existing(SYN_JSONL_RANDOM_CANDIDATES)
    synthetic_cluster_df = load_synth_jsonl(syn_cluster_path)
    synthetic_random_df  = load_synth_jsonl(syn_random_path)

    print("Embedding TRAIN sentences (one pass)...")
    full_embeddings = generate_embeddings(data["sentence"].tolist())

    # Map seed sets to indices within TRAIN only
    seed_idx_clustered = data.index[data["sentence"].isin(seed_cluster_df["sentence"])].tolist()
    seed_idx_random    = data.index[data["sentence"].isin(seed_random_df["sentence"])].tolist()
    print(f"Matched CLUSTERED seed indices in TRAIN: {len(seed_idx_clustered)}")
    print(f"Matched RANDOM seed indices in TRAIN:    {len(seed_idx_random)}")
    _warn_filtered(seed_cluster_df, seed_idx_clustered, "Clustered seeds")
    _warn_filtered(seed_random_df,  seed_idx_random,    "Random seeds")

    print("Computing fixed t-SNE layout on TRAIN data...")
    reduced = TSNE(n_components=2, random_state=SEED).fit_transform(full_embeddings)
    x_min, x_max = _expand_limits(reduced[:, 0].min(), reduced[:, 0].max())
    y_min, y_max = _expand_limits(reduced[:, 1].min(), reduced[:, 1].max())
    xlim, ylim = (x_min, x_max), (y_min, y_max)

    # Clustered (fixed layout)
    plot_tsne_projection_with_given_reduction(
        data, seed_idx_clustered, reduced, xlim, ylim,
        title_suffix="(Clustered Seeds)", outfile_suffix="_labeled_clustered_seed_fixed"
    )
    plot_tsne_projection_per_class_with_given_reduction(
        data, seed_idx_clustered, reduced, xlim, ylim,
        title_tag="Clustered Seeds", file_tag="_clustered_seed_fixed"
    )

    # Random (fixed layout)
    plot_tsne_projection_with_given_reduction(
        data, seed_idx_random, reduced, xlim, ylim,
        title_suffix="(Random Seeds)", outfile_suffix="_labeled_random_seed_fixed"
    )
    plot_tsne_projection_per_class_with_given_reduction(
        data, seed_idx_random, reduced, xlim, ylim,
        title_tag="Random Seeds", file_tag="_random_seed_fixed"
    )

    # Overlay (fixed layout)
    plot_tsne_overlay_two_seed_sets(
        data, reduced, xlim, ylim, seed_idx_random, seed_idx_clustered
    )

    # Optional: real vs synthetic (use TRAIN vs each synthetic set)
    plot_tsne_combined_real_synthetic(data, synthetic_cluster_df,
                                      outname="tsne_projection_real_vs_synthetic_clustered.png")
    plot_tsne_per_class_real_synthetic(data, synthetic_cluster_df, file_tag="_clustered")

    plot_tsne_combined_real_synthetic(data, synthetic_random_df,
                                      outname="tsne_projection_real_vs_synthetic_random.png")
    plot_tsne_per_class_real_synthetic(data, synthetic_random_df, file_tag="_random")

    print("Done.")
