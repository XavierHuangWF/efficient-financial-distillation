import openai
import pandas as pd
import json
import time
import numpy as np
from sklearn.metrics import classification_report
from datasets import load_dataset
from tqdm import tqdm

# ==== CONFIG ====
SEED = 24266
API_KEY = "*****"
SLEEP_TIME = 1.2
MODEL = "gpt-4o"
OUTPUT_PATH = "../outputs/chatgpt4o_test_predictions.json"
CLASS_REPORT_PATH = "../outputs/chatgpt4o_classification_report.json"

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)

set_seed(SEED)

# ==== Load Financial Phrasebank ====
print("Loading dataset...")
dataset = load_dataset("takala/financial_phrasebank", "sentences_allagree", split="train")
df = pd.DataFrame(dataset)

# ==== Manual Stratified Split ====
from sklearn.model_selection import train_test_split
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=SEED)

id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}
label2id = {v: k for k, v in id2label.items()}
test_df = test_df.reset_index(drop=True)

# ==== OpenAI Client ====
client = openai.OpenAI(api_key=API_KEY)

# ==== Prompt Template ====
def make_prompt(sentence):
    return f"""
You are a sentiment analysis assistant for financial news.
Classify the sentiment of the following financial news sentence into one of three categories:
- Negative
- Neutral
- Positive

Sentence: "{sentence}"

Respond with just one word: Negative, Neutral, or Positive.
"""

# ==== Inference ====
predictions = []

for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
    sentence = row["sentence"]
    true_label = row["label"]

    prompt = make_prompt(sentence)
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        pred_text = response.choices[0].message.content.strip().lower()

        if "negative" in pred_text:
            pred_label = 0
        elif "neutral" in pred_text:
            pred_label = 1
        elif "positive" in pred_text:
            pred_label = 2
        else:
            pred_label = -1  # error fallback

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

# ==== Evaluate ====
y_true = [label2id[x["true_label"]] for x in predictions if x["pred_label"] in label2id]
y_pred = [label2id[x["pred_label"]] for x in predictions if x["pred_label"] in label2id]

report = classification_report(y_true, y_pred, target_names=["Negative", "Neutral", "Positive"], output_dict=True)

with open(CLASS_REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2)

print("Classification Report:")
print(json.dumps(report, indent=2))
print(f"Saved classification report to {CLASS_REPORT_PATH}")
