import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

SEED = 24266

DATASET_CONFIG = {
    "phrasebank": {
        "labels": ["Negative", "Neutral", "Positive"],
        "default_input": "./outputs/seed_data.jsonl",
        "default_output": "./outputs/synthetic_data_from_Seed.jsonl",
    },
    "twitter": {
        "labels": ["Bearish", "Bullish", "Neutral"],
        "default_input": "./outputs/2seed_data_twitter_random.jsonl",
        "default_output": "./outputs/2synthetic_data_from_Seed_random.jsonl",
    },
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**32)


def load_jsonl(path: str) -> pd.DataFrame:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    df = pd.DataFrame(data)

    required = {"instruction", "input", "output"}
    if not required.issubset(df.columns):
        raise ValueError(f"JSONL missing required keys: {required}")
    return df


def template_1_dynamic(df: pd.DataFrame, target_label: str, labels: list[str], seed: int):
    examples = {}
    for sentiment in labels:
        sample_row = df[df["output"] == sentiment].sample(1, random_state=seed)
        examples[sentiment] = sample_row["input"].values[0]

    prompt = f"""
You are training a sentiment classification assistant for financial news.
Given the following instruction and examples, write a new financial news sentence with the same sentiment.

Instruction: Classify the sentiment of this financial news sentence.
Examples:
1. {examples[labels[0]]} → {labels[0]}
2. {examples[labels[1]]} → {labels[1]}
3. {examples[labels[2]]} → {labels[2]}

Now generate a new financial news sentence that expresses the following sentiment:
→ {target_label}
"""
    return examples, prompt.strip()


def template_2(seed_sentence: str, label: str) -> str:
    return f"""
The following sentence has a financial sentiment label of {label}:
{seed_sentence}

Generate 3 alternative sentences that:
- Express the same sentiment ({label})
- Remain realistic within the finance/news domain
- Use different wording or phrasing
""".strip()


def template_3(seed_list: list[str], label: str) -> str:
    seed_block = "\n".join([f"{i+1}. {s}" for i, s in enumerate(seed_list)])
    return f"""
Below are examples of {label} financial news:
{seed_block}

Generate 5 more realistic financial news sentences with the same sentiment ({label}).
""".strip()


def clean_generated_lines(text: str) -> list[str]:
    lines = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue

        # remove common bullet/number prefixes
        line = line.lstrip("-• ").strip()
        if len(line) > 1 and line[0].isdigit():
            parts = line.split(".", 1)
            if len(parts) == 2 and parts[0].isdigit():
                line = parts[1].strip()

        if line:
            lines.append(line)
    return lines


def generate_synthetic(
    dataset: str,
    model: str,
    input_jsonl: str,
    output_jsonl: str,
    sleep_time: float,
    seed: int,
) -> None:
    set_seed(seed)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)
    df = load_jsonl(input_jsonl)
    labels = DATASET_CONFIG[dataset]["labels"]

    outputs = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        sentence = row["input"]
        label = row["output"]

        try:
            # Template 1
            example_dict, prompt1 = template_1_dynamic(df, label, labels, seed)
            r1 = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt1}],
                temperature=0.7,
            )
            outputs.append(
                {
                    "instruction": "Classify the sentiment of this financial news sentence.",
                    "input": r1.choices[0].message.content.strip(),
                    "output": label,
                    "source": "template1",
                    "seed": example_dict,
                }
            )
            time.sleep(sleep_time)

            # Template 2
            prompt2 = template_2(sentence, label)
            r2 = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt2}],
                temperature=0.85,
            )
            for line in clean_generated_lines(r2.choices[0].message.content.strip()):
                outputs.append(
                    {
                        "instruction": "Classify the sentiment of this financial news sentence.",
                        "input": line,
                        "output": label,
                        "source": "template2",
                        "seed": sentence,
                    }
                )
            time.sleep(sleep_time)

            # Template 3
            row_seed = stable_hash(sentence)
            same_label_pool = df[(df["output"] == label) & (df["input"] != sentence)]

            if len(same_label_pool) < 2:
                continue

            other_seeds = same_label_pool.sample(
                n=2, random_state=row_seed
            )["input"].tolist()

            seed_list = [sentence] + other_seeds
            prompt3 = template_3(seed_list, label)
            r3 = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt3}],
                temperature=0.85,
            )
            for line in clean_generated_lines(r3.choices[0].message.content.strip()):
                outputs.append(
                    {
                        "instruction": "Classify the sentiment of this financial news sentence.",
                        "input": line,
                        "output": label,
                        "source": "template3",
                        "seed": seed_list,
                    }
                )
            time.sleep(sleep_time)

        except Exception as e:
            print(f"Error on sentence: {sentence[:80]}...")
            print(e)
            continue

    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(outputs)} synthetic examples to {output_jsonl}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["phrasebank", "twitter"], required=True)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--input-jsonl", default=None)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--sleep-time", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=24266)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = DATASET_CONFIG[args.dataset]

    input_jsonl = args.input_jsonl or cfg["default_input"]
    output_jsonl = args.output_jsonl or cfg["default_output"]

    generate_synthetic(
        dataset=args.dataset,
        model=args.model,
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        sleep_time=args.sleep_time,
        seed=args.seed,
    )