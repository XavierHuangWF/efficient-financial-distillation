# mcnemar_twitter.py
# ------------------------------------------------------------
# Compare a saved student model vs GPT-4o on the SAME Twitter
# Financial News test split using McNemar's test.
# - Rebuilds the exact 80/10/10 split with SEED=24266
# - Loads GPT-4o predictions from JSON you created
# - Runs inference for the student model from MODEL_DIR
# - Aligns items, computes b,c, p-values, 95% CI
# - Saves a JSON summary and a LaTeX table
# ------------------------------------------------------------

import os
import json
import numpy as np
import pandas as pd
import re
from math import fabs
from typing import Tuple

import torch
import torch.nn.functional as F

from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding,
)

# SciPy for tests + CI
from scipy.stats import chi2, binomtest
try:
    from scipy.stats import beta as beta_dist  # fallback for CP CI if needed
except Exception:
    beta_dist = None

# ----------------------------
# CONFIG
# ----------------------------
SEED = 24266
MODEL_DIR = "../outputs/best_model"     # <- path to your saved student model
GPT4O_JSON = "../outputs/chatgpt4o_twfn_predictions.json"
OUT_DIR = "../outputs"
STUDENT_NAME = "ModernBERT (distilled)"  # label for tables/prints
TEACHER_NAME = "GPT-4o"

# HF label mapping (0=Bearish, 1=Bullish, 2=Neutral)
LABEL_ID_BY_NAME = {"Bearish": 0, "Bullish": 1, "Neutral": 2}
LABEL_NAME_BY_ID = {v: k for k, v in LABEL_ID_BY_NAME.items()}
LABEL_NAMES = [LABEL_NAME_BY_ID[i] for i in range(3)]


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int = SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)


def load_twitter_df() -> pd.DataFrame:
    """Load zeroshot/twitter-financial-news-sentiment train+validation and clean."""
    ds_train = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train", trust_remote_code=True)
    ds_val   = load_dataset("zeroshot/twitter-financial-news-sentiment", split="validation", trust_remote_code=True)
    df = pd.concat([pd.DataFrame(ds_train), pd.DataFrame(ds_val)], ignore_index=True)

    # Normalize label column to 0/1/2
    if "label_text" in df.columns:
        lt = df["label_text"].str.strip().str.title()
        if not set(lt.unique()).issubset(set(LABEL_ID_BY_NAME.keys())):
            raise ValueError(f"Unexpected label_text values: {lt.unique()}")
        df["label"] = lt.map(LABEL_ID_BY_NAME).astype(int)
    else:
        if "label" not in df.columns:
            raise ValueError("Dataset missing 'label' column.")
        df["label"] = df["label"].astype(int)

    # Heuristic text column
    text_col = "sentence"
    if text_col not in df.columns:
        for c in ["text", "tweet", "content", "headline", "message", "document"]:
            if c in df.columns:
                text_col = c
                break
        if text_col not in df.columns:
            # fallback to first object column
            text_col = df.select_dtypes(include=["object"]).columns[0]

    if text_col != "sentence":
        df = df.rename(columns={text_col: "sentence"})
    df = df[["sentence", "label"]].copy()
    df["sentence"] = df["sentence"].astype(str).apply(clean_text)
    return df


def clean_text(text):
    if not isinstance(text, str):
        return text
    text = text.replace('\\\\', '\\')
    try:
        text = bytes(text, "utf-8").decode("unicode_escape")
    except Exception:
        pass
    text = re.sub(r"^\d+\.\s*", "", text)
    text = text.replace('"', "")
    return text.strip()


def stratified_split(df: pd.DataFrame, seed: int = SEED):
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=seed)
    val_df, test_df   = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_student_predictions(model_dir: str, test_df: pd.DataFrame) -> np.ndarray:
    """Run the saved student model on test_df and return y_pred (int ids)."""
    # Try to load tokenizer from saved dir, fallback to config._name_or_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except Exception:
        cfg_path = os.path.join(model_dir, "config.json")
        base_name = "answerdotai/ModernBERT-base"
        if os.path.exists(cfg_path):
            try:
                cfg = json.load(open(cfg_path, "r"))
                base_name = cfg.get("_name_or_path", base_name)
            except Exception:
                pass
        tokenizer = AutoTokenizer.from_pretrained(base_name)

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    hf_ds = Dataset.from_pandas(test_df.reset_index(drop=True))
    hf_ds = hf_ds.map(lambda ex: tokenizer(ex["sentence"], truncation=True), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(model=model, tokenizer=tokenizer, data_collator=data_collator)
    preds = trainer.predict(hf_ds).predictions
    y_pred = np.argmax(preds, axis=1)
    return y_pred


def load_gpt4o_predictions(json_path: str, n_items: int) -> np.ndarray:
    """Load GPT-4o JSON predictions and map to ids; -1 for invalid/unknown/error.
       Assumes JSON list is in the SAME order as the test_df iteration.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    def to_id(lbl: str) -> int:
        if lbl in LABEL_ID_BY_NAME:
            return LABEL_ID_BY_NAME[lbl]
        return -1

    arr = np.full(shape=(n_items,), fill_value=-1, dtype=int)
    m = min(n_items, len(preds))
    for i in range(m):
        lbl = preds[i].get("pred_label", "Unknown")
        arr[i] = to_id(lbl)
    return arr


def mcnemar_test(correct_A: np.ndarray, correct_B: np.ndarray) -> Tuple[int, int, float, float]:
    """Return (b, c, p_exact_or_cc, chi2_cc) where:
       b = A correct, B wrong; c = A wrong, B correct
       p = exact binomial (if b+c<25) else continuity-corrected chi2 p-value
       chi2_cc = continuity-corrected chi-square statistic
    """
    b = int(np.sum((correct_A == 1) & (correct_B == 0)))
    c = int(np.sum((correct_A == 0) & (correct_B == 1)))
    n_disc = b + c

    if n_disc == 0:
        return b, c, 1.0, 0.0

    # continuity-corrected chi-square
    chi2_cc = (fabs(b - c) - 1.0)**2 / n_disc

    if n_disc < 25:
        p = binomtest(k=min(b, c), n=n_disc, p=0.5, alternative="two-sided").pvalue
    else:
        p = 1 - chi2.cdf(chi2_cc, df=1)

    return b, c, float(p), float(chi2_cc)


def clopper_pearson_ci(b: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact CI for proportion p = b/n (discordant proportion)."""
    if n == 0:
        return (0.0, 1.0)
    try:
        # SciPy >= 1.9
        ci = binomtest(k=b, n=n, p=0.5).proportion_ci(confidence_level=1 - alpha, method="exact")
        return float(ci.low), float(ci.high)
    except Exception:
        if beta_dist is None:
            return (np.nan, np.nan)
        lower = beta_dist.ppf(alpha / 2, b, n - b + 1) if b > 0 else 0.0
        upper = beta_dist.ppf(1 - alpha / 2, b + 1, n - b) if b < n else 1.0
        return float(lower), float(upper)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Build the same Twitter test split
    full_df = load_twitter_df()
    _, _, test_df = stratified_split(full_df, seed=SEED)
    y_true = test_df["label"].to_numpy().astype(int)
    N = len(test_df)

    # 2) Student predictions (Model A)
    print(f"[Info] Loading student model from: {MODEL_DIR}")
    y_pred_A = load_student_predictions(MODEL_DIR, test_df)
    assert len(y_pred_A) == N, "Student predictions length mismatch."

    # 3) GPT-4o predictions (Model B)
    print(f"[Info] Loading GPT-4o predictions from: {GPT4O_JSON}")
    y_pred_B = load_gpt4o_predictions(GPT4O_JSON, n_items=N)

    # 4) Align on valid indices (discard items where GPT pred invalid)
    mask_valid = (y_pred_B >= 0)
    n_invalid = int(np.sum(~mask_valid))
    if n_invalid > 0:
        print(f"[Warn] Excluding {n_invalid} items with invalid GPT-4o predictions (Unknown/Error).")

    y_true_aln = y_true[mask_valid]
    yA_aln = y_pred_A[mask_valid]
    yB_aln = y_pred_B[mask_valid]

    # 5) Correctness flags
    correct_A = (yA_aln == y_true_aln).astype(int)
    correct_B = (yB_aln == y_true_aln).astype(int)

    # Accuracies (on aligned subset)
    acc_A = float(np.mean(correct_A)) if len(correct_A) > 0 else np.nan
    acc_B = float(np.mean(correct_B)) if len(correct_B) > 0 else np.nan

    # 6) McNemar
    b, c, p_val, chi2_cc = mcnemar_test(correct_A, correct_B)
    n_disc = b + c

    # 7) CI for discordant proportion b/(b+c)
    cp_low, cp_high = clopper_pearson_ci(b, n_disc, alpha=0.05)
    prop = (b / n_disc) if n_disc > 0 else np.nan

    # 8) Print summary
    print("\n================ McNemar Test (Twitter test set) ================")
    print(f"Aligned test items: {len(y_true_aln)} (dropped {n_invalid} invalid GPT-4o preds)")
    print(f"Accuracy {STUDENT_NAME}: {acc_A:.4f}")
    print(f"Accuracy {TEACHER_NAME} : {acc_B:.4f}")
    print("\nDiscordant pairs:")
    print(f"  b = A correct, B wrong : {b}")
    print(f"  c = A wrong,  B correct: {c}")
    print(f"  n_disc = b + c         : {n_disc}")
    if n_disc > 0:
        print(f"  b/(b+c)                : {prop:.4f} (95% CI [{cp_low:.4f}, {cp_high:.4f}])")
    print(f"\nContinuity-corrected chi^2: {chi2_cc:.4f}  (df=1)")
    print(f"McNemar p-value           : {p_val:.6f}")
    print("Interpretation: p < 0.05 indicates a significant difference in error patterns.\n")

    # 9) Save JSON summary
    summary = {
        "aligned_items": int(len(y_true_aln)),
        "dropped_invalid_gpt4o": int(n_invalid),
        "accuracy": {
            STUDENT_NAME: acc_A,
            TEACHER_NAME: acc_B
        },
        "discordant": {
            "b_A_correct_B_wrong": b,
            "c_A_wrong_B_correct": c,
            "n_disc": n_disc,
            "b_over_bplusc": prop,
            "ci_95_b_over_bplusc": [cp_low, cp_high]
        },
        "mcnemar": {
            "chi2_cc": chi2_cc,
            "p_value": p_val,
            "method": "exact binomial if n_disc<25 else continuity-corrected chi-square"
        },
        "labels": LABEL_NAMES,
        "student_model_dir": MODEL_DIR,
        "teacher_predictions_json": GPT4O_JSON,
    }
    json_path = os.path.join(OUT_DIR, "mcnemar_twitter_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[Saved] {json_path}")

    # 10) Save a small LaTeX table (2x2 + p-value)
    a = int(np.sum((correct_A == 1) & (correct_B == 1)))
    d = int(np.sum((correct_A == 0) & (correct_B == 0)))
    tex = rf"""
\begin{table}[t]
\centering
\caption{{McNemar comparison on Twitter test set ({STUDENT_NAME} vs {TEACHER_NAME}).}}
\label{{tab:mcnemar_twitter}}
\begin{tabular}{{lcc}}
\toprule
 & \textbf{{{TEACHER_NAME} Correct}} & \textbf{{{TEACHER_NAME} Wrong}} \\
\midrule
\textbf{{{STUDENT_NAME} Correct}} & {a} & {b} \\
\textbf{{{STUDENT_NAME} Wrong}}  & {c} & {d} \\
\midrule
\multicolumn{{3}}{{l}}{{Aligned items: {len(y_true_aln)}, discordant $b+c={n_disc}$; $p={p_val:.4f}$}}\\
\bottomrule
\end{tabular}
\end{table}
""".strip("\n")
    tex_path = os.path.join(OUT_DIR, "mcnemar_twitter_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"[Saved] {tex_path}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("GPU device:", torch.cuda.get_device_name(0))
    main()
