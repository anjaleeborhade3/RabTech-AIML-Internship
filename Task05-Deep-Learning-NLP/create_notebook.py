from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Task 05: Deep Learning / NLP Spam Classifier
**RabTech Academy AI/ML Internship**

Build a text spam classifier using TF-IDF and a PyTorch neural network.
Includes dropout, batch normalization, early stopping, training curves, evaluation metrics, and confidence-based inference.

**Responsible use:** This is an educational demo. Predictions can be wrong; do not use it as the sole basis for important decisions."""))

cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import copy
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PROJECT_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch:", torch.__version__)
print("Device:", device)"""))

cells.append(nbf.v4.new_markdown_cell("""## 1. Prepare text dataset

Use two public text categories from the 20 Newsgroups dataset as a reproducible binary text-classification exercise: `rec.sport.hockey` and `sci.space`. This demonstrates the NLP pipeline, although the classes are topics rather than actual spam/ham labels."""))

cells.append(nbf.v4.new_code_cell("""categories = ["rec.sport.hockey", "sci.space"]
train_raw = fetch_20newsgroups(
    subset="train", categories=categories,
    remove=("headers", "footers", "quotes"), random_state=SEED
)
test_raw = fetch_20newsgroups(
    subset="test", categories=categories,
    remove=("headers", "footers", "quotes"), random_state=SEED
)

texts = list(train_raw.data) + list(test_raw.data)
labels = np.concatenate([train_raw.target, test_raw.target])
# Keep only non-empty messages
valid = np.array([bool(str(t).strip()) for t in texts])
texts = np.array(texts, dtype=object)[valid].tolist()
labels = labels[valid]

# Reproducible split; vectorizer is fitted only on training text.
X_train_text, X_test_text, y_train, y_test = train_test_split(
    texts, labels, test_size=0.20, random_state=SEED, stratify=labels
)
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2), min_df=2)
X_train = vectorizer.fit_transform(X_train_text).toarray().astype(np.float32)
X_test = vectorizer.transform(X_test_text).toarray().astype(np.float32)

print("Training samples:", len(y_train))
print("Test samples:", len(y_test))
print("Vocabulary size:", len(vectorizer.vocabulary_))
print("Class names:", train_raw.target_names)"""))

cells.append(nbf.v4.new_markdown_cell("""## 2. Create validation set and neural network

The network uses Linear layers, Batch Normalization, ReLU and Dropout. Early stopping monitors validation loss to help limit overfitting."""))

cells.append(nbf.v4.new_code_cell("""X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.20, random_state=SEED, stratify=y_train
)

def make_loader(X, y, batch_size=64, shuffle=False):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(np.asarray(y, dtype=np.int64)))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader = make_loader(X_val, y_val)
test_loader = make_loader(X_test, y_test)

class TextClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.network(x)

model = TextClassifier(X_train.shape[1]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
print(model)"""))

cells.append(nbf.v4.new_markdown_cell("""## 3. Train with early stopping

Training stops when validation loss does not improve for several epochs. The best validation-loss model weights are restored."""))

cells.append(nbf.v4.new_code_cell("""MAX_EPOCHS = 30
PATIENCE = 4
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_loss = float("inf")
best_state = copy.deepcopy(model.state_dict())
patience_counter = 0

for epoch in range(MAX_EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(yb)
        correct += (logits.argmax(1) == yb).sum().item()
        total += len(yb)

    train_loss = running_loss / total
    train_acc = correct / total

    model.eval()
    val_loss_sum, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            val_loss_sum += loss.item() * len(yb)
            val_correct += (logits.argmax(1) == yb).sum().item()
            val_total += len(yb)

    val_loss = val_loss_sum / val_total
    val_acc = val_correct / val_total
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    print(f"Epoch {epoch+1:02d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

model.load_state_dict(best_state)
print("Restored best validation-loss model.")"""))

cells.append(nbf.v4.new_markdown_cell("## 4. Plot training and validation curves"))

cells.append(nbf.v4.new_code_cell("""epochs = range(1, len(history["train_loss"]) + 1)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(epochs, history["train_loss"], marker="o", label="Train loss")
axes[0].plot(epochs, history["val_loss"], marker="o", label="Validation loss")
axes[0].set_title("Loss Convergence")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross-entropy loss")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(epochs, history["train_acc"], marker="o", label="Train accuracy")
axes[1].plot(epochs, history["val_acc"], marker="o", label="Validation accuracy")
axes[1].set_title("Accuracy Convergence")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(REPORT_DIR / "training_curves.png", dpi=150)
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("## 5. Evaluate on unseen test data"))

cells.append(nbf.v4.new_code_cell("""model.eval()
all_probs, all_preds, all_true = [], [], []
with torch.no_grad():
    for xb, yb in test_loader:
        logits = model(xb.to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(probs.argmax(axis=1))
        all_true.extend(yb.numpy())

test_accuracy = accuracy_score(all_true, all_preds)
metrics = {
    "Accuracy": test_accuracy,
    "Precision (macro)": precision_score(all_true, all_preds, average="macro", zero_division=0),
    "Recall (macro)": recall_score(all_true, all_preds, average="macro", zero_division=0),
    "F1 (macro)": f1_score(all_true, all_preds, average="macro", zero_division=0),
}
display(pd.DataFrame([metrics]).round(4))
print(classification_report(all_true, all_preds, target_names=train_raw.target_names, zero_division=0))
ConfusionMatrixDisplay.from_predictions(all_true, all_preds, display_labels=train_raw.target_names, xticks_rotation=30)
plt.title("Test Confusion Matrix")
plt.tight_layout()
plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150)
plt.show()

pd.DataFrame([metrics]).to_csv(REPORT_DIR / "test_metrics.csv", index=False)"""))

cells.append(nbf.v4.new_markdown_cell("""## 6. Save model and inference assets

The model weights and fitted TF-IDF vectorizer are saved separately. To reproduce inference, use the same vectorizer and feature dimension."""))

cells.append(nbf.v4.new_code_cell("""import joblib

torch.save({
    "state_dict": model.cpu().state_dict(),
    "input_dim": int(X_train.shape[1]),
    "class_names": list(train_raw.target_names),
}, MODEL_DIR / "text_classifier.pt")
joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib")
print("Saved:", MODEL_DIR / "text_classifier.pt")
print("Saved:", MODEL_DIR / "tfidf_vectorizer.joblib")
model.to(device)"""))

cells.append(nbf.v4.new_markdown_cell("## 7. Sample inference with confidence scores"))

cells.append(nbf.v4.new_code_cell("""def predict_text(message):
    features = vectorizer.transform([message]).toarray().astype(np.float32)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(features).to(device))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
    index = int(np.argmax(probabilities))
    return {
        "message": message,
        "predicted_category": train_raw.target_names[index],
        "confidence": float(probabilities[index]),
        "class_probabilities": {
            train_raw.target_names[i]: float(probabilities[i]) for i in range(len(probabilities))
        }
    }

samples = [
    "The team played a great hockey game and scored in the final period.",
    "NASA scientists discussed a new mission to explore Mars and space."
]
display(pd.DataFrame([predict_text(s) for s in samples]))"""))

cells.append(nbf.v4.new_markdown_cell("""## Limitations and responsible use

- The two categories are topic labels, not spam/ham labels. This is an NLP classification demonstration, not a spam detector.
- Removing headers, footers and quotes reduces metadata leakage, but text datasets can still contain biases and duplicates.
- TF-IDF ignores some semantic and contextual meaning.
- Confidence scores are model outputs and may be miscalibrated; they are not guarantees.
- Performance on these categories does not imply performance on other domains."""))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"}
}
out = Path("notebooks/nlp_text_classifier.ipynb")
nbf.write(nb, out)
print("Notebook created:", out)
