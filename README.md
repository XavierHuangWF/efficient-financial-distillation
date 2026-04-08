# efficient-financial-distillation

Code for the LREC 2026 paper:

**Efficient Financial Language Understanding via Distillation with Synthetic Data**

This repository contains code for a low-resource financial sentiment classification pipeline built around:

- seed example selection
- synthetic data generation with prompt templates
- teacher-model evaluation
- compact student-model training
- result analysis and visualization

The project studies whether carefully selected seed examples and LLM-generated synthetic data can improve compact student models on financial sentiment tasks under limited real-data settings.

---

## Broader research relevance

Although the experiments in this repository focus on financial sentiment analysis, the underlying methodology is intended for broader low-resource and privacy-constrained AI settings. In particular, the framework for clustering-based seed selection, synthetic data generation, and compact model distillation is relevant to domains where labeled data are limited, deployment efficiency matters, and reliable decision support is important, including healthcare monitoring and critical infrastructure applications.

---

## How to run

Below are example commands for the main pipeline stages.

### 1. Synthetic data generation

```bash
python scripts/generation/generate_synthetic.py --dataset phrasebank
python scripts/generation/generate_synthetic.py --dataset twitter
```

### 2. GPT-4o evaluation

```bash
python scripts/generation/chatgpt4o_eval.py.py --dataset phrasebank
python scripts/generation/chatgpt4o_eval.py.py --dataset twitter
```

### 3. Seed selection

```bash
python scripts/seed_selection/SaveSeed_financial_phrasebank.py
python scripts/seed_selection/SaveRandomSeed_financial_phrasebank.py
python scripts/seed_selection/SaveSeed_twitter-financial-news.py
python scripts/seed_selection/SavedandomSeed_twitter-financial-news.py
```

### 4. Seed distance visualization

```bash
python scripts/seed_selection/visualize_seed_distances.py
python scripts/seed_selection/visualize_Randomseed_distances.py
```

### 5. Student model training

PhraseBank:

```bash
python scripts/training/Distilbert_phrasebank_trainer.py --mode full
python scripts/training/Modernbert_phrasebank_trainer.py --mode full
python scripts/training/Tinybert_phrasebank_trainer.py --mode full
```

Twitter Financial News:

```bash
python scripts/training/Distilbert_twitter_trainer.py --mode full
python scripts/training/Modernbert_twitter_trainer.py --mode full
python scripts/training/Tinybert_twitter_trainer.py --mode full
```

Examples with synthetic + seed data:

```bash
python scripts/training/Distilbert_phrasebank_trainer.py --mode synthetic --seed-source jsonl --seed-jsonl ../outputs/seed_data.jsonl --synthetic-jsonl ../outputs/synthetic_data_from_Seed.jsonl --use-early-stopping
python scripts/training/Modernbert_twitter_trainer.py --mode synthetic --seed-source jsonl --seed-jsonl ../outputs/seed_data.jsonl --synthetic-jsonl ../outputs/synthetic_data_from_Seed.jsonl --use-early-stopping
python scripts/training/Tinybert_twitter_trainer.py --mode synthetic --seed-source jsonl --seed-jsonl ../outputs/seed_data.jsonl --synthetic-jsonl ../outputs/synthetic_data_from_Seed.jsonl --use-early-stopping
```

### 6. Evaluation and comparison

```bash
python scripts/evaluation/FinBert_test.py
python scripts/evaluation/test_phrasebank_plot.py
python scripts/evaluation/test_twitter_plot.py
python scripts/evaluation/test_phrasebank_modernBERT.py
python scripts/evaluation/test_twitterNews_modernBERT.py
python scripts/evaluation/Plot_comparison.py
python scripts/evaluation/mcnemar_twitter.py
```

---

## Installation

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

If you use OpenAI-based generation or evaluation, store your API key as an environment variable instead of hardcoding it.

### Windows CMD

```bash
set OPENAI_API_KEY=your_key_here
```

### PowerShell

```bash
$env:OPENAI_API_KEY="your_key_here"
```

### macOS / Linux

```bash
export OPENAI_API_KEY=your_key_here
```

---

## Tasks and datasets

This repository currently focuses on two financial sentiment datasets:

- Financial PhraseBank
- Twitter Financial News Sentiment

---

## Method overview

The pipeline is organized into the following stages:

1. **Seed selection**  
   Select representative seed examples using clustering-based sampling or random sampling.

2. **Synthetic data generation**  
   Use prompt templates and an LLM to generate additional sentiment-labeled financial text from seed examples.

3. **Teacher / reference model evaluation**  
   Evaluate strong pretrained or API-based models such as FinBERT or GPT-4o.

4. **Student model training**  
   Train compact models such as DistilBERT, TinyBERT, and ModernBERT on:
   - full supervised data
   - real seed data only
   - synthetic + seed merged data
   - standard-seed and random-seed variants

5. **Evaluation and comparison**  
   Compare performance across datasets, model families, and settings using metrics, plots, and statistical tests.

---

## Repository structure

```text
efficient-financial-distillation/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ data/
│  ├─ interim/
│  ├─ processed/
│  └─ raw/
├─ Demo/
│  └─ app.py
├─ docs/
├─ literature/
├─ outputs/
│  └─ .gitkeep
├─ prompts/
│  ├─ .gitkeep
│  ├─ template1.txt
│  ├─ template2.txt
│  └─ template3.txt
├─ scripts/
│  ├─ evaluation/
│  │  ├─ FinBert_test.py
│  │  ├─ mcnemar_twitter.py
│  │  ├─ Plot_comparison.py
│  │  ├─ test_phrasebank_modernBERT.py
│  │  ├─ test_phrasebank_plot.py
│  │  ├─ test_twitter_plot.py
│  │  └─ test_twitterNews_modernBERT.py
│  ├─ generation/
│  │  ├─ chatgpt4o_eval.py.py
│  │  └─ generate_synthetic.py
│  ├─ seed_selection/
│  │  ├─ SavedandomSeed_twitter-financial-news.py
│  │  ├─ SaveRandomSeed_financial_phrasebank.py
│  │  ├─ SaveSeed_financial_phrasebank.py
│  │  ├─ SaveSeed_twitter-financial-news.py
│  │  ├─ visualize_Randomseed_distances.py
│  │  └─ visualize_seed_distances.py
│  └─ training/
│     ├─ Distilbert_phrasebank_trainer.py
│     ├─ Distilbert_twitter_trainer.py
│     ├─ Modernbert_phrasebank_trainer.py
│     ├─ Modernbert_twitter_trainer.py
│     ├─ Tinybert_phrasebank_trainer.py
│     └─ Tinybert_twitter_trainer.py
```

---

## Notes

This repository currently contains a cleaner training structure than before. The main model training code has been consolidated into six trainer scripts:

- DistilBERT PhraseBank
- DistilBERT Twitter
- ModernBERT PhraseBank
- ModernBERT Twitter
- TinyBERT PhraseBank
- TinyBERT Twitter

Legacy evaluation, seed-selection, and plotting scripts are still kept as separate files for reproducibility and comparison.

---

## Citation

If you use this repository, please cite the associated paper.

```bibtex
@inproceedings{huang2026efficient,
  title={Efficient Financial Language Understanding via Distillation with Synthetic Data},
  author={Huang, Wen-Fong and Simpson, Edwin},
  booktitle={Proceedings of the 2026 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2026)},
  year={2026}
}
```

---

## License

This project is released under the MIT License.
