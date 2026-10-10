# Responsible Data Card: Customer Churn Dataset

## 1. Dataset purpose
This dataset is used for educational exploration of customer churn prediction.
It must not be used alone to make consequential decisions about customers.
Any outreach should be optional, respectful, and subject to human review.

## 2. Provenance and permission
Source file: `customer-churn-training.csv`
Source URL: https://rabtechacademy.in/starter-files/aiml/customer-churn-training.csv

The dataset was supplied as a starter file by RabTech Academy. The underlying
data-generation method, original data owner, collection date, consent status,
license, and permission for commercial use are not documented in the supplied
file or brief. These details must be confirmed with the provider before any
real-world or commercial use. Do not assume that customer consent is established.

## 3. Population and representation
The file contains 12 customer records. The represented population, sampling
method, collection period, geography, and representativeness are unknown.
The small sample may not reflect real customer populations or churn patterns.
Sensitive attributes are not explicitly listed, but proxy effects have not
been assessed.

## 4. Features and target
- `customer_id`: customer identifier; exclude from model training.
- `tenure_months`: tenure in months, inferred from the column name.
- `support_tickets`: support ticket count, inferred from the column name.
- `monthly_spend_inr`: monthly spend in INR, inferred from the column name.
- `last_login_days`: days since last login, inferred from the column name.
- `plan_type`: plan category.
- `churned`: target label; 1 means churned and 0 means not churned,
  based on the dataset values and task context.

Feature definitions and label timing should be confirmed with the provider.
Potential leakage risk exists if any feature was measured after the churn
outcome or would not be available at prediction time.

## 5. Quality checks
- Rows: 12
- Columns: 7
- Missing values: 0 in all columns
- Exact duplicate rows: 0
- Target counts: 7 not churned (0), 5 churned (1)
- Plan counts: Basic 5, Standard 4, Pro 3

No missing values or duplicate rows were observed in the initial checks.
Outliers, label correctness, temporal consistency, and data-entry validity
have not been independently verified. With only 12 rows, a reliable
train/test estimate is not possible; avoid claims of generalizable accuracy.

## 6. Privacy and safeguards
Customer IDs may still be linkable to individuals in the original system.
Do not publish identifiable customer information. Limit access to the data,
avoid attempting re-identification, and remove identifiers from model inputs.
Confirm the provider's permission before redistributing the dataset.

## 7. Risks and mitigations
- False positive: unnecessary retention outreach. Mitigation: optional,
  respectful contact and human review.
- False negative: a customer who needs support may be missed. Mitigation:
  preserve normal support channels; never rely only on the model.
- Bias or proxy effects: model behavior may differ across groups not represented
  in the file. Mitigation: collect appropriate, lawfully permitted evaluation
  data before deployment and assess group performance where justified.
- Leakage: features may reveal information from after the outcome. Mitigation:
  define a prediction timestamp and verify feature availability.
- Overfitting: 12 records are insufficient for reliable generalization.
  Mitigation: use the model for learning only; obtain a larger representative
  dataset before deployment.
- Misuse: predictions could be used to penalize customers. Mitigation: restrict
  use to supportive service improvement and document prohibited uses.

## 8. Intended evaluation
- Baseline: majority-class predictor.
- Exploratory metrics: confusion matrix, precision, recall, F1-score, and
  balanced accuracy, with the small-sample limitation clearly stated.
- Calibration: do not claim calibrated probabilities from this tiny dataset;
  calibration needs substantially more representative data.
- Fairness: sensitive-group analysis is not possible from the available fields.
  Do not infer sensitive attributes.
- Error analysis: inspect each prediction cautiously and document uncertainty.

## 9. Abstention, monitoring, and rollback
Return "needs human review" for missing, invalid, or out-of-range inputs and
for low-confidence predictions. Before any deployment, define monitoring for
input drift, prediction rates, error rates, complaints, and harmful disparities.
Suspend the model and revert to human-led processes if quality or safety checks fail.

## 10. Overall assessment
This is a small educational starter dataset. Provenance, consent, license,
sampling, and feature timing remain unverified. It is not suitable on its own
for production decisions or claims about real customer behavior.
