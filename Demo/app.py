import os
import sys
import json
import torch
import torch.nn.functional as F
import pandas as pd
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABEL_NAMES = ["Negative", "Neutral", "Positive"]

def resource_path(relative_path: str) -> str:
    # Works for normal Python, PyInstaller one-dir, and PyInstaller one-file
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

MODEL_DIR = resource_path("best_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer exactly like your eval logic
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
except Exception:
    fallback = "answerdotai/ModernBERT-base"
    try:
        with open(os.path.join(MODEL_DIR, "config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        fallback = cfg.get("_name_or_path", fallback)
    except Exception:
        pass
    tokenizer = AutoTokenizer.from_pretrained(fallback)

# Load the same saved fine-tuned model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)
model.to(device)
model.eval()

@torch.no_grad()
def predict_sentiment(text):
    text = (text or "").strip()
    if not text:
        empty_df = pd.DataFrame({
            "label": LABEL_NAMES,
            "probability": [0.0, 0.0, 0.0]
        })
        return "Please enter a sentence.", None, empty_df

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()

    pred_id = int(probs.argmax())
    pred_label = LABEL_NAMES[pred_id]
    pred_conf = float(probs[pred_id])

    label_scores = {
        LABEL_NAMES[i]: float(probs[i]) for i in range(len(LABEL_NAMES))
    }

    df = pd.DataFrame({
        "label": LABEL_NAMES,
        "probability": [float(x) for x in probs]
    })

    summary = f"Predicted sentiment: {pred_label} (confidence: {pred_conf:.3f})"
    return summary, label_scores, df

examples = [
    ["The company raised its full-year guidance after strong quarterly earnings."],
    ["The bank reported weaker-than-expected profit and cut its outlook."],
    ["Revenue remained broadly unchanged from the previous year."],
]

with gr.Blocks(title="Financial Sentiment Demo") as demo:
    gr.Markdown("## Financial Sentiment Classifier")
    gr.Markdown("Type a financial sentence or headline to get sentiment prediction.")

    text_input = gr.Textbox(
        label="Input text",
        lines=4,
        placeholder="Enter a financial sentence or headline..."
    )

    with gr.Row():
        predict_btn = gr.Button("Predict")
        clear_btn = gr.Button("Clear")

    summary_output = gr.Textbox(label="Prediction")
    label_output = gr.Label(label="Class probabilities", num_top_classes=3)
    table_output = gr.Dataframe(label="Probability table")

    gr.Examples(examples=examples, inputs=text_input)

    predict_btn.click(
        fn=predict_sentiment,
        inputs=text_input,
        outputs=[summary_output, label_output, table_output]
    )

    text_input.submit(
        fn=predict_sentiment,
        inputs=text_input,
        outputs=[summary_output, label_output, table_output]
    )

    clear_btn.click(
        fn=lambda: ("", None, pd.DataFrame({
            "label": LABEL_NAMES,
            "probability": [0.0, 0.0, 0.0]
        })),
        inputs=None,
        outputs=[summary_output, label_output, table_output]
    )

if __name__ == "__main__":
    demo.launch(
        inbrowser=True,
        share=False,
        server_name="127.0.0.1",
        server_port=7860
    )