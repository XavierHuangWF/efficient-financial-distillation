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

## How to run

Below are example commands for the main pipeline stages.

### 1. Synthetic data generation

```bash
python scripts/generation/generate_synthetic.py --dataset phrasebank
python scripts/generation/generate_synthetic.py --dataset twitter
```

### 2. Seed selection

```bash
python scripts/seed_selection/select_seeds.py --dataset phrasebank --method clustered
python scripts/seed_selection/select_seeds.py --dataset phrasebank --method random
python scripts/seed_selection/select_seeds.py --dataset twitter --method clustered
python scripts/seed_selection/select_seeds.py --dataset twitter --method random
```

### 3. Student model training

```bash
python scripts/training/train.py --model distilbert --dataset phrasebank --data-mode real
python scripts/training/train.py --model tinybert --dataset phrasebank --data-mode synthetic
python scripts/training/train.py --model modernbert --dataset twitter --data-mode synthetic
```

### 4. Evaluation

```bash
python scripts/evaluation/evaluate.py --dataset phrasebank --model distilbert
python scripts/evaluation/plot_results.py
python scripts/evaluation/stat_tests.py
```

---

## Installation

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

If you use OpenAI-based generation, store your API key as an environment variable instead of hardcoding it.

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
   Evaluate strong pretrained or API-based models such as FinBERT or GPT-based classifiers.

4. **Student model training**  
   Train compact models such as DistilBERT, TinyBERT, and ModernBERT on:
   - real seed data only
   - synthetic-augmented data
   - clustered-seed and random-seed variants

5. **Evaluation and comparison**  
   Compare performance across datasets, model families, and settings using metrics, plots, and statistical tests.

---

## Repository structure

```text
efficient-financial-distillation/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ requirements.txt
├─ prompts/
│  ├─ template1.txt
│  ├─ template2.txt
│  └─ template3.txt
├─ scripts/
│  ├─ generation/
│  │  ├─ generate_synthetic.py
│  │  ├─ chatgpt4o_test_phrasebank.py
│  │  └─ chatgpt4o_test_twitter_financial_news.py
│  ├─ seed_selection/
│  │  ├─ select_seeds.py
│  │  ├─ visualize_seed_distances.py
│  │  └─ visualize_randomseed_distances.py
│  ├─ training/
│  │  ├─ train.py
│  │  └─ callbacks.py
│  ├─ evaluation/
│  │  ├─ evaluate.py
│  │  ├─ plot_results.py
│  │  └─ stat_tests.py
│  └─ misc/
├─ demo/
│  └─ app.py
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ sample/
├─ results/
│  └─ .gitkeep
└─ docs/
   └─ appendix/
```

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
