import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.cells = [
    nbf.v4.new_markdown_cell("""# RabTech AI/ML Internship — Task 03
## Feature Engineering & Preprocessing Pipeline

**Dataset:** Titanic passenger dataset  
**Target:** `survived` (0 = did not survive, 1 = survived)

### Objectives
- Inspect missing values and feature types.
- Split the data before fitting preprocessing transformations.
- Impute missing values, scale numeric features, and one-hot encode categorical features.
- Use a reusable Scikit-Learn `ColumnTransformer` and `Pipeline`.
- Analyse correlations and model feature importance.

**Leakage precautions:** The target and duplicate/outcome-derived columns such as `alive` are excluded from the input features. All imputers, scalers, and encoders are fitted within the training pipeline only."""),
    nbf.v4.new_markdown_cell("## 1. Import libraries and load data"),
    nbf.v4.new_code_cell("""from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display

DATA_PATH = Path("../data/titanic.csv")
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
display(df.head())
print("\\\\nColumn types:")
display(df.dtypes.to_frame("dtype"))"""),
    nbf.v4.new_markdown_cell("## 2. Data quality and missing values"),
    nbf.v4.new_code_cell("""print("Missing values by column:")
missing = df.isna().sum().sort_values(ascending=False)
display(missing[missing > 0].to_frame("missing_count"))

print("Duplicate rows:", df.duplicated().sum())
print("\\\\nTarget distribution:")
display(df["survived"].value_counts().sort_index().to_frame("count"))

plt.figure(figsize=(9, 4))
missing[missing > 0].plot(kind="bar")
plt.title("Missing Values by Column")
plt.ylabel("Number of missing rows")
plt.tight_layout()
plt.show()"""),
    nbf.v4.new_markdown_cell("""## 3. Select target and features

We exclude `alive`, which directly restates the survival outcome, to avoid target leakage. We also exclude `class` because it duplicates the information in `pclass`, and `embark_town` because it duplicates `embarked`. These choices reduce redundant information.

The dataset is used for educational analysis; it is not a recommendation system for real-world passenger decisions."""),
    nbf.v4.new_code_cell("""target = "survived"
drop_columns = ["survived", "alive", "class", "embark_town"]
X = df.drop(columns=drop_columns)
y = df[target].astype(int)

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)
print("Total features:", X.shape[1])"""),
    nbf.v4.new_markdown_cell("""## 4. Train/test split BEFORE preprocessing

The split is performed before any imputer, scaler, or encoder is fitted. The test set is held out for final evaluation."""),
    nbf.v4.new_code_cell("""from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training target proportions:")
display(y_train.value_counts(normalize=True).rename("proportion").to_frame())"""),
    nbf.v4.new_markdown_cell("## 5. Build preprocessing pipelines"),
    nbf.v4.new_code_cell("""from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ))
])

print("Reusable preprocessing and model pipeline created.")"""),
    nbf.v4.new_markdown_cell("## 6. Fit on training data and evaluate on test data"),
    nbf.v4.new_code_cell("""from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, ConfusionMatrixDisplay
)

model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 3))
print("Recall:", round(recall_score(y_test, y_pred, zero_division=0), 3))
print("F1-score:", round(f1_score(y_test, y_pred, zero_division=0), 3))
print("\\\\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Test Set Confusion Matrix")
plt.tight_layout()
plt.show()"""),
    nbf.v4.new_markdown_cell("""## 7. Correlation analysis

Pearson correlation is calculated for numeric columns only. Correlation describes association and does not prove causation. The plot is exploratory and does not use the held-out test labels to fit the model."""),
    nbf.v4.new_code_cell("""numeric_for_corr = df.select_dtypes(include="number").drop(
    columns=["survived"], errors="ignore"
)
corr = numeric_for_corr.corr()

plt.figure(figsize=(9, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Numeric Feature Correlation Heatmap")
plt.tight_layout()
plt.show()"""),
    nbf.v4.new_markdown_cell("""## 8. Tree-based feature importance

Feature importance is extracted from the fitted Random Forest. One-hot encoded categories are shown as separate transformed features. Importance values are model-specific and should not be interpreted as causal effects."""),
    nbf.v4.new_code_cell("""fitted_preprocessor = model_pipeline.named_steps["preprocessor"]
fitted_model = model_pipeline.named_steps["model"]

feature_names = fitted_preprocessor.get_feature_names_out()
importances = fitted_model.feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

display(importance_df.head(15))

plt.figure(figsize=(10, 6))
top_features = importance_df.head(15).sort_values("importance")
plt.barh(top_features["feature"], top_features["importance"])
plt.title("Top 15 Random Forest Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()"""),
    nbf.v4.new_markdown_cell("""## 9. Reusable inference example

The fitted pipeline accepts raw feature rows and applies the same imputation, scaling, encoding, and model steps. It can also handle unseen categories because `OneHotEncoder` uses `handle_unknown="ignore"`."""),
    nbf.v4.new_code_cell("""example_rows = X_test.head(3)
display(example_rows)

example_predictions = model_pipeline.predict(example_rows)
print("Predictions:", example_predictions.tolist())
print("0 = did not survive; 1 = survived")"""),
    nbf.v4.new_markdown_cell("""## 10. Conclusions and limitations

- A reusable `ColumnTransformer` and `Pipeline` combine preprocessing with the classifier.
- Train/test splitting happens before fitting preprocessing transformations.
- Missing numerical values are median-imputed; missing categorical values are most-frequent-imputed.
- Numerical features are standardized; categorical features are one-hot encoded.
- Outcome-derived `alive` and redundant columns are excluded.
- Correlation and feature importance are exploratory, not causal.
- A single train/test split can vary; the reported test metrics are not proof of generalization. Further validation and careful model review would be needed for any real application.""")
]

nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {"name": "python", "version": "3"}
}

nbf.write(nb, "notebooks/feature_engineering_pipeline.ipynb")
print("Created notebooks/feature_engineering_pipeline.ipynb")
