import numpy as np
import torch
import evaluate
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SEED = 24266
MODEL_NAME = "ProsusAI/finbert"  # FinBERT baseline

def DataLoad():
    ds = load_dataset("takala/financial_phrasebank", "sentences_allagree", split="train", trust_remote_code=True)
    return ds  # HuggingFace Dataset

def Stratified_Split(ds):
    # convert to arrays for splitting
    labels = np.array(ds["label"])
    idx = np.arange(len(ds))

    train_idx, temp_idx = train_test_split(idx, test_size=0.2, stratify=labels, random_state=SEED)
    temp_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=temp_labels, random_state=SEED)

    return train_idx, val_idx, test_idx

def batched_predict(model, tokenizer, texts, batch_size=64, max_length=128, device="cpu"):
    model.eval()
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        preds.extend(logits.argmax(dim=-1).detach().cpu().numpy().tolist())
    return np.array(preds, dtype=int)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = DataLoad()

    _, _, test_idx = Stratified_Split(ds)
    test_texts = [ds[i]["sentence"] for i in test_idx]
    y_true = np.array([ds[i]["label"] for i in test_idx], dtype=int)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)

    # Map FinBERT model label IDs -> PhraseBank label IDs (0 neg, 1 neu, 2 pos)
    id2label = model.config.id2label  # e.g. {0:'negative',1:'neutral',2:'positive'}
    finbert_str2id = {"negative": 0, "neutral": 1, "positive": 2}
    remap = {int(k): finbert_str2id[str(v).lower()] for k, v in id2label.items()}

    raw_pred = batched_predict(model, tokenizer, test_texts, batch_size=64, max_length=128, device=device)
    y_pred = np.array([remap[int(i)] for i in raw_pred], dtype=int)

    acc = evaluate.load("accuracy").compute(predictions=y_pred, references=y_true)["accuracy"]
    f1 = evaluate.load("f1").compute(predictions=y_pred, references=y_true, average="macro")["f1"]

    print(f"FinBERT Test Accuracy: {acc:.4f}")
    print(f"FinBERT Test Macro-F1: {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["Negative", "Neutral", "Positive"], digits=4))

if __name__ == "__main__":
    main()
