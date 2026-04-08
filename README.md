# efficient-financial-distillation

Code for the LREC 2026 paper:

**Efficient Financial Language Understanding via Distillation with Synthetic Data**

![method_pipeline.png](docs/method_pipeline.png)

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

## Workflow overview

The workflow implemented in this repository follows the pipeline shown above:

1. **Embedding-based clustering for seed selection**  
   Select representative examples from the original dataset using clustering-based or random seed selection.

2. **Teacher prompting and synthetic generation**  
   Use prompt templates and teacher models to generate additional labeled financial text from the selected seeds.

3. **Data ingestion and cleaning**  
   Normalize text, clean labels, merge seed and synthetic data when needed, and prepare train/validation/test splits.

4. **Tokenization**  
   Convert text into model-ready inputs for compact student models.

5. **Student training and distillation**  
   Train compact models such as DistilBERT, TinyBERT, and ModernBERT under full-data, seed-only, or synthetic-plus-seed settings.

6. **Evaluation**  
   Compare models and settings using test metrics, plots, and statistical comparison scripts.

---

## Quick start

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

PhraseBank examples:

```bash
python scripts/training/Distilbert_phrasebank_trainer.py --mode full
python scripts/training/Modernbert_phrasebank_trainer.py --mode full
python scripts/training/Tinybert_phrasebank_trainer.py --mode full
```

Twitter Financial News examples:

```bash
python scripts/training/Distilbert_twitter_trainer.py --mode full
python scripts/training/Modernbert_twitter_trainer.py --mode full
python scripts/training/Tinybert_twitter_trainer.py --mode full
```

Synthetic + seed examples:

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

Create and activate a Python environment, then install the packages used by the current scripts:

```bash
pip install torch transformers datasets evaluate sentence-transformers scikit-learn pandas numpy matplotlib openai tqdm
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

## Repository structure

```text
efficient-financial-distillation/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ Demo/
│  └─ app.py
├─ docs/
│  └─ method_pipeline.png
├─ literature/
├─ outputs/
├─ prompts/
│  ├─ template1.txt
│  ├─ template2.txt
│  └─ template3.txt
└─ scripts/
   ├─ evaluation/
   │  ├─ FinBert_test.py
   │  ├─ mcnemar_twitter.py
   │  ├─ Plot_comparison.py
   │  ├─ test_phrasebank_modernBERT.py
   │  ├─ test_phrasebank_plot.py
   │  ├─ test_twitter_plot.py
   │  └─ test_twitterNews_modernBERT.py
   ├─ generation/
   │  ├─ chatgpt4o_eval.py.py
   │  └─ generate_synthetic.py
   ├─ seed_selection/
   │  ├─ SavedandomSeed_twitter-financial-news.py
   │  ├─ SaveRandomSeed_financial_phrasebank.py
   │  ├─ SaveSeed_financial_phrasebank.py
   │  ├─ SaveSeed_twitter-financial-news.py
   │  ├─ visualize_Randomseed_distances.py
   │  └─ visualize_seed_distances.py
   └─ training/
      ├─ Distilbert_phrasebank_trainer.py
      ├─ Distilbert_twitter_trainer.py
      ├─ Modernbert_phrasebank_trainer.py
      ├─ Modernbert_twitter_trainer.py
      ├─ Tinybert_phrasebank_trainer.py
      └─ Tinybert_twitter_trainer.py
```

---

## Notes

This repository now uses a cleaner training structure than the earlier experiment-specific layout. The main training code has been consolidated into six trainer scripts:

- DistilBERT PhraseBank
- DistilBERT Twitter
- ModernBERT PhraseBank
- ModernBERT Twitter
- TinyBERT PhraseBank
- TinyBERT Twitter

Legacy evaluation, seed-selection, and plotting scripts are still kept as separate files for reproducibility and comparison.

Some filenames in the current repository still reflect legacy naming, and the commands above intentionally match the current filenames exactly.

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
