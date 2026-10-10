import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Task 04: Supervised Modeling & Hyperparameter Tuning

**RabTech Academy AI/ML Internship**

This notebook compares four classification models, performs hyperparameter tuning using stratified 5-fold GridSearchCV, evaluates a held-out test set, plots ROC curves and confusion matrices, and saves the champion model.

**Dataset:** Scikit-Learn Breast Cancer Wisconsin dataset  
**Positive class:** Malignant (cancer); recall for this class is especially important.  
**Disclaimer:** Educational demonstration only—not for clinical diagnosis."""))

cells.append(nbf.v4.new_code_cell("""import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

PROJECT_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_PATH = PROJECT_DIR / "data" / "breast_cancer.csv"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print("Missing values:", int(df.isna().sum().sum()))
display(df.head())"""))

cells.append(nbf.v4.new_markdown_cell("""## 1. Define target and split data

The original dataset encodes malignant as 0 and benign as 1. We convert the target so **1 = malignant** and **0 = benign**. The test set is held out before model tuning."""))

cells.append(nbf.v4.new_code_cell("""X = df.drop(columns="target")
y = (df["target"] == 0).astype(int)  # 1 = malignant, 0 = benign

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
print("Training malignant rate:", round(y_train.mean(), 3))
print("Test malignant rate:", round(y_test.mean(), 3))"""))

cells.append(nbf.v4.new_markdown_cell("""## 2. Define four models and tuning grids

Scaling is fitted inside each pipeline and therefore only learned from each training fold. Stratified 5-fold cross-validation is used for hyperparameter selection."""))

cells.append(nbf.v4.new_code_cell("""models_and_grids = {
    "Logistic Regression": (
        LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE),
        {"model__C": [0.1, 1, 10]}
    ),
    "Random Forest": (
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        {"model__n_estimators": [100, 200],
         "model__max_depth": [None, 10],
         "model__min_samples_split": [2, 5]}
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {"model__n_estimators": [100, 200],
         "model__learning_rate": [0.05, 0.1],
         "model__max_depth": [1, 3]}
    ),
    "Support Vector Machine": (
        SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE),
        {"model__C": [0.1, 1, 10],
         "model__kernel": ["linear", "rbf"]}
    )
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
searches = {}
comparison_rows = []

for name, (estimator, param_grid) in models_and_grids.items():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", estimator)
    ])
    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False
    )
    search.fit(X_train, y_train)
    searches[name] = search

    pred = search.predict(X_test)
    prob = search.predict_proba(X_test)[:, 1]

    comparison_rows.append({
        "Model": name,
        "Best CV ROC-AUC": search.best_score_,
        "Test Accuracy": accuracy_score(y_test, pred),
        "Test Precision": precision_score(y_test, pred, zero_division=0),
        "Test Recall": recall_score(y_test, pred, zero_division=0),
        "Test F1": f1_score(y_test, pred, zero_division=0),
        "Test ROC-AUC": roc_auc_score(y_test, prob),
        "Best Parameters": str(search.best_params_)
    })
    print(f"Finished: {name} | Best CV ROC-AUC: {search.best_score_:.4f}")

comparison = pd.DataFrame(comparison_rows).sort_values(
    "Best CV ROC-AUC", ascending=False
).reset_index(drop=True)

display(comparison)
comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)"""))

cells.append(nbf.v4.new_markdown_cell("""## 3. Compare ROC-AUC curves

Curves are calculated only on the held-out test set. The tuning and champion selection are based on training cross-validation ROC-AUC."""))

cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(9, 7))

for name, search in searches.items():
    prob = search.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Malignant Recall)")
plt.title("ROC Curves — Held-out Test Set")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(REPORT_DIR / "roc_curves.png", dpi=150)
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("""## 4. Champion model and confusion matrix

The champion is selected using the highest **cross-validation ROC-AUC**, not by repeatedly choosing based on test results. Review recall and false negatives carefully because the positive class is malignant."""))

cells.append(nbf.v4.new_code_cell("""champion_name = comparison.loc[0, "Model"]
champion_search = searches[champion_name]
champion = champion_search.best_estimator_

print("Champion:", champion_name)
print("Best CV ROC-AUC:", round(champion_search.best_score_, 4))
print("Best parameters:", champion_search.best_params_)

champion_pred = champion.predict(X_test)
champion_prob = champion.predict_proba(X_test)[:, 1]

print("\\nClassification report (1 = malignant):")
print(classification_report(
    y_test, champion_pred,
    labels=[0, 1],
    target_names=["Benign", "Malignant"],
    zero_division=0
))

cm = confusion_matrix(y_test, champion_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malignant"])
disp.plot(cmap="Blues", values_format="d")
plt.title(f"Champion Confusion Matrix: {champion_name}")
plt.tight_layout()
plt.savefig(REPORT_DIR / "champion_confusion_matrix.png", dpi=150)
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("""## 5. Save reusable model and metadata

The saved pipeline includes its fitted scaler and classifier. Input features must use the same column names and order as the training data."""))

cells.append(nbf.v4.new_code_cell("""model_path = MODEL_DIR / "champion_model.joblib"
joblib.dump(champion, model_path)

metadata = {
    "champion_model": champion_name,
    "positive_class": "malignant",
    "selection_metric": "5-fold cross-validation ROC-AUC",
    "best_cv_roc_auc": float(champion_search.best_score_),
    "test_roc_auc": float(roc_auc_score(y_test, champion_prob)),
    "feature_columns": list(X.columns),
    "best_parameters": {k: str(v) for k, v in champion_search.best_params_.items()},
    "intended_use": "Educational demonstration only; not for clinical diagnosis."
}
joblib.dump(metadata, MODEL_DIR / "champion_metadata.joblib")

print("Saved model:", model_path)
print("Saved metadata: models/champion_metadata.joblib")"""))

cells.append(nbf.v4.new_markdown_cell("""## 6. Example inference

This demonstrates using the saved pipeline on a held-out sample. Scores are model outputs, not clinical probabilities or medical advice."""))

cells.append(nbf.v4.new_code_cell("""loaded_model = joblib.load(MODEL_DIR / "champion_model.joblib")
sample = X_test.iloc[[0]]
malignant_probability = loaded_model.predict_proba(sample)[0, 1]
prediction = int(loaded_model.predict(sample)[0])

print("Predicted class:", "Malignant" if prediction == 1 else "Benign")
print(f"Model score for malignant class: {malignant_probability:.3f}")
display(sample)"""))

cells.append(nbf.v4.new_markdown_cell("""## Limitations and responsible use

- This is a small, pre-cleaned educational dataset; performance may not generalize.
- Test metrics are estimates from one held-out split.
- Hyperparameter selection uses training cross-validation; the test set is for final evaluation.
- False negatives and false positives have different consequences. Clinical decisions require validated tools and qualified professionals.
- This notebook is not a medical device and must not be used to diagnose patients."""))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {"name": "python", "version": "3"}
}

with open("notebooks/model_comparison_and_tuning.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook created: notebooks/model_comparison_and_tuning.ipynb")
