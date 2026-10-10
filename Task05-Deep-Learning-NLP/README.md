# Task 05: Deep Learning / NLP Text Classifier

## Overview
This project demonstrates text classification using a PyTorch neural network and TF-IDF features.

## Dataset
The project uses the Scikit-learn 20 Newsgroups dataset with two topic categories:
- `rec.sport.hockey`
- `sci.space`

This is a topic-classification demonstration, not a spam/ham classifier.

## Features
- Text preprocessing and TF-IDF vectorization
- PyTorch neural network with Batch Normalization and Dropout
- Validation-based early stopping
- Training and validation loss/accuracy curves
- Classification metrics and confusion matrix
- Sample predictions with confidence scores
- Saved model weights and TF-IDF vectorizer

## Run the project
Activate the virtual environment and install dependencies:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Execute the notebook:

```bash
jupyter nbconvert --to notebook --execute \
  notebooks/nlp_text_classifier.ipynb \
  --output nlp_text_classifier_executed.ipynb
```

The first run may require an internet connection to download the dataset.

## Output files
- `models/text_classifier.pt` — trained PyTorch model
- `models/tfidf_vectorizer.joblib` — fitted TF-IDF vectorizer
- `reports/` — training curves and evaluation plots

## Responsible use
This is an educational demonstration. It is not intended for production moderation or other high-stakes decisions.
