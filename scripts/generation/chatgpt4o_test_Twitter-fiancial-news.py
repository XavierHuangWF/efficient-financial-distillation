import os
import openai
import pandas as pd
import json
import time
import numpy as np
from sklearn.metrics import classification_report
from datasets import load_dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import re

# ==== CONFIG ====
SEED = 24266
# Prefer env var; rotate your key if it was exposed.
API_KEY = "*****"
SLEEP_TIME = 1.2
MODEL = "gpt-4o"
OUTPUT_PATH = "../outputs/chatgpt4o_twfn_predictions.json"
CLASS_REPORT_PATH = "../outputs/chatgpt4o_twfn_classification_report.json"

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)

set_seed(SEED)

# ==== Load zeroshot/twitter-financial-news-sentiment and concat train+validation ====
print("Loading dataset...")
ds_train = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
ds_val   = load_dataset("zeroshot/twitter-financial-news-sentiment", split="validation")
df = pd.concat([pd.DataFrame(ds_train), pd.DataFrame(ds_val)], ignore_index=True)

# Standardize to 'sentence' text column
if "text" in df.columns and "sentence" not in df.columns:
    df = df.rename(columns={"text": "sentence"})
elif "sentence" not in df.columns:
    # Fallback: pick a string column
    text_col = next((c for c in df.columns if df[c].dtype == object), None)
    if text_col is None:
        raise ValueError("Could not find a text column.")
    df = df.rename(columns={text_col: "sentence"})

# Ensure labels are ints 0/1/2: 0=Bearish, 1=Bullish, 2=Neutral
df["label"] = df["label"].astype(int)

# ==== Stratified 80/10/10 split ====
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
val_df,   test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=SEED)
test_df = test_df.reset_index(drop=True)

id2label = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
label2id = {v: k for k, v in id2label.items()}

# ==== OpenAI Client ====
client = openai.OpenAI(api_key=API_KEY)

# ==== Prompt Template (Bearish/Bullish/Neutral) ====
def make_prompt(sentence: str) -> str:
    return f"""
You are a sentiment analysis assistant for financial news.
Classify the sentiment of the following sentence into exactly one of:
- Bearish
- Bullish
- Neutral

Sentence: "{sentence}"

Respond with just one word: Bearish, Bullish, or Neutral.
"""

# Robust parser: normalize common variants to our three labels
def parse_label(text: str) -> int:
    t = text.strip().lower()
    # Remove punctuation/quotes
    t = re.sub(r'[^a-z]+', ' ', t).strip()
    # direct matches
    if "bearish" in t:
        return 0
    if "bullish" in t:
        return 1
    if "neutral" in t:
        return 2
    # common fallbacks
    if "negative" in t or "sell" in t or "pessim" in t:
        return 0
    if "positive" in t or "buy" in t or "optim" in t:
        return 1
    return -1

# ==== Inference ====
predictions = []

print("Running inference on test split...")
for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
    sentence = row["sentence"]
    true_label = int(row["label"])

    prompt = make_prompt(sentence)
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        pred_text = response.choices[0].message.content.strip()
        pred_label = parse_label(pred_text)

        predictions.append({
            "sentence": sentence,
            "true_label": id2label[true_label],
            "pred_label": id2label.get(pred_label, "Unknown"),
            "raw_response": pred_text
        })

        time.sleep(SLEEP_TIME)

    except Exception as e:
        print("Error:", e)
        predictions.append({
            "sentence": sentence,
            "true_label": id2label[true_label],
            "pred_label": "Error",
            "raw_response": str(e)
        })
        time.sleep(SLEEP_TIME)

# ==== Save predictions ====
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(predictions, f, indent=2, ensure_ascii=False)
print(f"Saved predictions to {OUTPUT_PATH}")

# ==== Evaluate (filter Unknown/Error) ====
valid = [p for p in predictions if p["pred_label"] in label2id]
y_true = [label2id[p["true_label"]] for p in valid]
y_pred = [label2id[p["pred_label"]] for p in valid]

report = classification_report(
    y_true, y_pred,
    target_names=[id2label[0], id2label[1], id2label[2]],
    output_dict=True
)

with open(CLASS_REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("Classification Report:")
print(json.dumps(report, indent=2))
print(f"Saved classification report to {CLASS_REPORT_PATH}")
