# ESG text classifier

Classifies sentences from corporate sustainability reports as **Environmental**, **Social**, **Governance** or **non-ESG**.

The project compares two approaches on the same test set: a classic machine learning baseline (TF-IDF + logistic regression) and a fine-tuned transformer (DistilBERT).

## Project objectives

1. Build a simple, explainable baseline for 4-class ESG sentence classification.
2. Fine-tune DistilBERT on the same data and measure how much it improves on the baseline.
3. Analyze where each model fails and why (error analysis).
4. Deploy the best model as a live demo where a user pastes report text and sees each sentence labeled.

## Project status

| Stage | Model | Status | Test macro-F1 |
|---|---|---|---|
| 1 | TF-IDF + logistic regression (baseline) | Done | `TODO: from results/baseline_metrics.json` |
| 2 | DistilBERT (fine-tuned) | Not started | n/a |
| 3 | Error analysis and comparison | Not started | n/a |
| 4 | Live demo (Gradio on Hugging Face Spaces) | Not started | n/a |

## Dataset

- **Source:** [FinanceMTEB/ESG](https://huggingface.co/datasets/FinanceMTEB/ESG) on Hugging Face.
- **Content:** single sentences taken from company annual and sustainability reports.
- **Labels:** `environmental`, `social`, `governance`, `non-esg`.
- **Size after cleaning:** `TODO: number of rows printed in section 2 of the notebook` sentences.

Class distribution:

| Class | Sentences | Share |
|---|---|---|
| environmental | `TODO` | `TODO` % |
| social | `TODO` | `TODO` % |
| governance | `TODO` | `TODO` % |
| non-esg | `TODO` | `TODO` % |

![Class counts](results/class_counts.png)

The 95th percentile sentence length is `TODO` words. This guides the `max_length` setting for DistilBERT.

## Preprocessing

The steps below are in `notebooks/01_eda_and_baseline.ipynb`.

1. Load all splits of the dataset and combine them into one table.
2. Strip leading and trailing whitespace and drop empty rows.
3. Remove exact duplicate sentences, so the same sentence cannot appear in both train and test (data leakage).
4. Split into train (80%), validation (10%) and test (10%), stratified by label so every split has the same class proportions.
5. Fix the random seed (42) so the splits are reproducible. Both models are scored on the same test set.

No stemming, stop-word removal or other text normalization is applied. TF-IDF lowercases the text. DistilBERT will use its own tokenizer.

## Stage 1: baseline (done)

**Model:** TF-IDF features with a logistic regression classifier (scikit-learn).

**Settings tried**, selected by validation macro-F1:

- n-gram range: unigrams, or unigrams and bigrams
- regularization strength `C`: 0.1, 1, 10
- fixed: `min_df=2`, `sublinear_tf=True`, `class_weight="balanced"`

**Best setting:** `TODO: ngram_range and C from results/baseline_metrics.json`

**Results on the test set:**

| Metric | Score |
|---|---|
| Macro-F1 | `TODO` |
| Accuracy | `TODO` |

Per-class F1:

| Class | F1 |
|---|---|
| environmental | `TODO` |
| social | `TODO` |
| governance | `TODO` |
| non-esg | `TODO` |

![Baseline confusion matrix](results/baseline_confusion_matrix.png)

**Why macro-F1:** it averages the F1 score of each class, so every class counts equally even when the classes are different sizes. Accuracy alone can look good while a small class is predicted badly.

**Early finding:** some sentences fit both Social and Governance. For example, sentences about anti-corruption training for employees are labeled `governance`, while similar sentences about compliance or code-of-conduct training are labeled `social`. This label inconsistency limits how well any model can separate the two classes.

## Stage 2: DistilBERT (not started)

Planned approach:

- Fine-tune `distilbert-base-uncased` with a 4-label classification head.
- Train on the same `train` split, select the best checkpoint on `validation`, and report on the same `test` split as the baseline.
- Report macro-F1, per-class F1, a confusion matrix, training time and inference speed.

Results will be added here and to the status table above.

## Repository structure

```
esg-text-classifier/
├── notebooks/
│   └── 01_eda_and_baseline.ipynb    EDA, splits and the TF-IDF baseline
├── results/
│   ├── baseline_metrics.json        baseline scores and settings
│   ├── baseline_confusion_matrix.png
│   ├── baseline_errors.csv          test sentences the baseline got wrong
│   ├── class_counts.png
│   └── sentence_lengths.png
├── LICENSE
└── README.md
```

## How to run

1. Open `notebooks/01_eda_and_baseline.ipynb` in [Google Colab](https://colab.research.google.com).
2. Run all cells from top to bottom. No GPU is needed for the baseline.
3. The notebook creates a `data/` folder with the splits and a `results/` folder with the metrics and plots.

## Limitations

- The dataset is in English and comes from large company reports, so the model may be less accurate on other document types or regions.
- Each sentence has exactly one label, although some sentences cover more than one ESG pillar.
- The model identifies what a sentence is about. It does not judge the quality of a company's disclosure.

## License

Code is released under the MIT License. The dataset belongs to its original authors; see the dataset page for its terms.
