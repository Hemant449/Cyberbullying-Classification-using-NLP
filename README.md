# Cyberbullying Text Classification with NLP

This project investigates a focused five-class text-classification task for social-media moderation using classical NLP and an LSTM. It also documents the original six-class problem and the evidence used to choose a narrower scope.

## Project motivation

The source dataset contains six labels: religion, age, ethnicity, gender, `not_cyberbullying`, and `other_cyberbullying`. The final task focuses on four identity-targeted categories (religion, age, gender, and ethnicity) plus non-bullying content. This is a deliberate project-scope decision; it is **not** a claim that all examples in the excluded category are harmless, illegal, or incorrectly labelled.

The six-class models, confusion matrices, lexical similarity, and optional sentence embeddings help diagnose class difficulty. Such results are supporting evidence, not proof of label invalidity. A blinded human label audit is included as a way to collect direct evidence about label overlap.

## Repository structure

```text
cyberbullying-classification-nlp/
├── README.md
├── requirements.txt
├── requirements-optional-similarity.txt
├── .gitignore
├── data/
│   └── README.md
└── notebooks/
    ├── 01_problem_framing_and_six_class_diagnostics.ipynb
    ├── 02_five_class_tfidf_logistic_regression.ipynb
    └── 03_five_class_lstm.ipynb
```

## Notebooks

| Notebook | Purpose |
|---|---|
| `01_problem_framing_and_six_class_diagnostics.ipynb` | Dataset quality, shared split, six-class Naive Bayes and Logistic Regression, class-wise errors, TF-IDF centroid similarity, optional sentence embeddings, blinded review export. |
| `02_five_class_tfidf_logistic_regression.ipynb` | Five-class classical NLP workflow, Naive Bayes baseline, validation-based selection of Logistic Regression `C`, interpretability, error analysis, and a matched six-class control on the same retained test examples. |
| `03_five_class_lstm.ipynb` | Sequence preprocessing, train-only tokenizer, post-padding, masked learned embeddings, LSTM training, per-class evaluation, training curves, and an optional matched six-class LSTM control. |

## Dataset setup

The CSV is **not included**. Obtain it from the source used for your project and check the dataset's terms before redistributing it. Put the file here:

```text
data/cyberbullying_tweets.csv
```

The notebooks also search for `cyberbullying_tweets.csv` in the current directory and `/content/` for Colab. Expected columns are either `tweet_text` and `cyberbullying_type`, or `text` and `label`.

## Run order

1. Install `requirements.txt` from the repository root.
2. Open and run Notebook 01 to inspect the six-class problem and diagnostics.
3. Run Notebook 02 for the main TF-IDF + Logistic Regression model.
4. Run Notebook 03 for the main LSTM model.

All three notebooks include the same canonical duplicate/conflict handling and 70/15/15 split from the original six-class rows. They print a split fingerprint; for the same CSV it should match across notebooks. The five-class notebooks filter out `other_cyberbullying` only after this shared split, preserving the same test rows for the retained classes.

TF-IDF vocabulary and IDF are fitted only on training data. The LSTM tokenizer is also fitted only on training data. Random oversampling is applied only to training data. Validation is used for small hyperparameter selection/early stopping; the test partition is reserved for final evaluation.

The sentence-embedding analysis in Notebook 01 is optional and disabled by default. Install `requirements-optional-similarity.txt` and set `RUN_SENTENCE_EMBEDDING_DIAGNOSTIC = True` to enable it. The six-class LSTM control in Notebook 03 is also optional because training a second recurrent network takes additional time.

## Results: how to report them responsibly

Earlier exploratory runs from this project recorded approximately:

| Model / task | Accuracy | Macro-F1 |
|---|---:|---:|
| Six-class Naive Bayes | 0.78 | 0.73 |
| Six-class Logistic Regression | 0.84 | 0.82 |
| Six-class LSTM | 0.84 | 0.82 |
| Five-class Naive Bayes | 0.85 | 0.83 |
| Five-class Logistic Regression | 0.92 | 0.91 |
| Five-class LSTM | 0.926 | 0.93 |

**These are historical exploratory results, not a controlled final benchmark.** The earlier runs used different splits and/or preprocessing rules. Re-run the cleaned notebooks and use the matched comparisons inside Notebooks 02 and 03 before presenting a direct five-class versus six-class performance comparison. Do not copy these numbers as the results of the revised notebooks without rerunning them.

## Methodological notes

- The source dataset's labels define the task; they do not independently establish whether a tweet is a crime or determine a moderation action.
- Cosine similarity measures similarity in the selected vector space. It does not prove semantic equivalence.
- Removing a class makes a task narrower and can improve measured scores; that improvement alone does not show that the removed class is invalid.
- The human audit exports a blinded review sheet and a private key. Do not commit the private key, completed review files, or raw tweet examples to a public repository.
- Sentiment/abuse language may be context-dependent. Manually inspect errors rather than asserting that a model prediction proves a source label wrong.

## Skills demonstrated

Text preprocessing, class-distribution and data-quality analysis, leakage-conscious splitting, TF-IDF, Naive Bayes, Logistic Regression, model interpretation, sentence embeddings, tokenization, padding and masking, LSTM architecture, oversampling, macro-F1, confusion-matrix analysis, error inspection, and experimental design.
