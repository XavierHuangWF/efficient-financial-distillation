import openai
import pandas as pd
import time
import json
from tqdm import tqdm
import numpy as np
import hashlib


# ==== CONFIG ====
SEED = 24266

client = openai.OpenAI(api_key="*****")  # Replace with your actual key
MODEL = "gpt-4o"
INPUT_JSONL = "../outputs/seed_data.jsonl"
OUTPUT_JSONL = "../outputs/synthetic_data_from_Seed.jsonl"
SLEEP_TIME = 1.2

# ==== SEED SETUP ====
def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
set_seed()

def stable_hash(text):
    return int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16) % (2**32)

# ==== LOAD JSONL ====
data = []
with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line.strip()))
df = pd.DataFrame(data)

# ==== TEMPLATES ====
def template_1_dynamic(df, target_label):
    examples = {}
    for sentiment in ["Negative", "Neutral", "Positive"]:
        sample_row = df[df["output"] == sentiment].sample(1, random_state=SEED)
        examples[sentiment] = sample_row["input"].values[0]
    return examples, f"""
You are training a sentiment classification assistant for financial news.
Given this instruction and a few labeled examples, write a new financial news sentence with the same sentiment:

Instruction: Classify the sentiment of this financial news sentence.
Examples:
1. {examples['Negative']} → Negative
2. {examples['Neutral']} → Neutral
3. {examples['Positive']} → Positive

Now generate a new financial news sentence that expresses the following sentiment:
→ {target_label}
"""

def template_2(seed_sentence, label):
    return f"""
The following sentence has a financial sentiment label of {label}:
{seed_sentence}

Generate 3 alternative sentences that:
- Express the same sentiment ({label})
- Remain realistic within the finance/news domain
- Use different wording or phrasing
"""

def template_3(seed_list, label):
    seed_block = "\n".join([f"{i+1}. {s}" for i, s in enumerate(seed_list)])
    return f"""
Below are examples of {label} financial news:
{seed_block}

Generate 5 more realistic financial news sentences with the same sentiment ({label}).
"""

# ==== GENERATION ====
outputs = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    sentence = row['input']
    label = row['output']

    try:
        # Template 1
        example_dict, prompt1 = template_1_dynamic(df, label)
        r1 = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt1}],
            temperature=0.7
        )
        outputs.append({
            "instruction": "Classify the sentiment of this financial news sentence.",
            "input": r1.choices[0].message.content.strip(),
            "output": label,
            "source": "template1",
            "seed": example_dict
        })
        time.sleep(SLEEP_TIME)

        # Template 2
        prompt2 = template_2(sentence, label)
        r2 = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt2}],
            temperature=0.85
        )
        for line in r2.choices[0].message.content.strip().split("\n"):
            if line.strip():
                outputs.append({
                    "instruction": "Classify the sentiment of this financial news sentence.",
                    "input": line.strip(),
                    "output": label,
                    "source": "template2",
                    "seed": sentence
                })
        time.sleep(SLEEP_TIME)

        # Ensure we don't re-sample the current sentence
        row_seed = stable_hash(sentence)
        other_seeds = df[(df['output'] == label) & (df['input'] != sentence)].sample(n=2, random_state=row_seed)[
            'input'].tolist()

        seed_list = [sentence] + other_seeds
        prompt3 = template_3(seed_list, label)
        r3 = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt3}],
            temperature=0.85
        )
        for line in r3.choices[0].message.content.strip().split("\n"):
            if line.strip():
                outputs.append({
                    "instruction": "Classify the sentiment of this financial news sentence.",
                    "input": line.strip(),
                    "output": label,
                    "source": "template3",
                    "seed": seed_list
                })
        time.sleep(SLEEP_TIME)

    except Exception as e:
        print("Error:", e)
        continue

# ==== SAVE JSONL ====
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for item in outputs:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f" Saved {len(outputs)} synthetic examples to {OUTPUT_JSONL}")
