# RabTech AI/ML Internship — Task 03

## Feature Engineering & Preprocessing Pipeline

This project demonstrates a reusable Scikit-Learn preprocessing pipeline using the Titanic dataset.

### Implemented
- Missing-value analysis
- Train/test split before preprocessing
- Numerical imputation and scaling
- Categorical imputation and One-Hot Encoding
- ColumnTransformer and Pipeline
- Random Forest classification
- Correlation heatmap and feature importance

### Dataset
https://github.com/mwaskom/seaborn-data

### Run
Install dependencies: python -m pip install pandas scikit-learn matplotlib seaborn jupyter ipykernel nbformat

Execute notebook: jupyter nbconvert --to notebook --execute notebooks/feature_engineering_pipeline.ipynb --output feature_engineering_pipeline_executed.ipynb

### Limitations
This is an educational project. Metrics depend on the train/test split. Feature importance does not establish causation.
