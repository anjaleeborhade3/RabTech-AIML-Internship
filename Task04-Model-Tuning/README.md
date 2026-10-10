# Task 04: Supervised Modeling & Hyperparameter Tuning

RabTech Academy AI/ML Internship

## Objective
Compare supervised classification models, tune hyperparameters, evaluate test performance, and save the best model.

## Dataset
Scikit-learn Breast Cancer Wisconsin dataset:
569 rows and 30 features.

Target:
- 1 = Malignant
- 0 = Benign

Educational use only. Not for clinical diagnosis.

## Models
- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)

## Techniques
- Train/test split
- StandardScaler inside an sklearn Pipeline
- Stratified 5-fold GridSearchCV
- ROC-AUC-based model selection
- Accuracy, precision, recall, F1-score and ROC-AUC
- ROC curves and confusion matrix
- Model serialization using Joblib

## Project Structure
- notebooks/ — modeling notebook
- models/ — saved model and metadata
- reports/ — metrics and evaluation plots
- requirements.txt — dependencies

## Run
Install dependencies:

pip install -r requirements.txt

Execute the notebook:

jupyter nbconvert --to notebook --execute notebooks/model_comparison_and_tuning.ipynb --output model_comparison_and_tuning_executed.ipynb

## Limitations
This is an educational exercise on a small, pre-cleaned dataset. Results may not generalize to real-world patients. It must not be used for diagnosis or treatment decisions.
