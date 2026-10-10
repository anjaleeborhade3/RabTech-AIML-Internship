# ML Problem-Framing Memo: Customer Churn Prediction

## 1. Problem and decision
The goal is to estimate whether a customer may discontinue a service.
The prediction may help a customer-support team decide which customers
could be offered optional support or asked for feedback.

The model must not automatically cancel services, deny benefits, increase
prices, or make other consequential decisions about a customer.

## 2. Prediction target
- Target column: `churned`
- `1`: customer churned
- `0`: customer did not churn
- Unit of observation: one customer record
- Prediction horizon: not specified in the supplied dataset; this must be
  defined before real-world deployment.

## 3. Available features
- `tenure_months`: length of customer tenure in months
- `support_tickets`: number of support tickets
- `monthly_spend_inr`: monthly spend in INR
- `last_login_days`: days since last login
- `plan_type`: subscription plan
- `customer_id`: identifier; excluded from model features

Feature definitions are inferred from column names and should be confirmed
with the dataset provider.

## 4. Stakeholders
- Customers
- Customer-support staff
- Customer-success or retention team
- Data/ML team
- Organization responsible for the service

## 5. Is machine learning justified?
Not yet for production. The dataset has only 12 records. A simple rule-based
approach and ordinary customer feedback should be considered as baselines.
ML may be explored as an educational prototype, but a larger representative
dataset and prospective validation are needed before operational use.

## 6. Non-ML baseline
Use a simple majority-class baseline that predicts the most frequent label
in the training data. Compare any model against this baseline. A real
retention team should also compare against existing support practices.

## 7. Metrics and error costs
Report confusion matrix, precision, recall, F1-score, and balanced accuracy
where appropriate. Use stratified evaluation only when sample size allows.
False positives may waste support time or annoy customers with unwanted
outreach. False negatives may miss customers who need assistance.
Choose an operating threshold only after stakeholders agree on these costs.

## 8. Abstention and human review
If data are missing, inputs are outside the training range, or model
confidence is insufficient, return "needs human review" rather than a
confident prediction. Staff should verify the situation and contact
customers respectfully.

## 9. Monitoring and rollback
Monitor input quality, class balance, prediction rates, error rates, and
customer complaints. Suspend the model if data quality changes, performance
degrades, harmful disparities appear, or predictions are used outside the
approved purpose. Revert to existing human-led support practices.

## 10. Limitations
The dataset contains only 12 rows and has no supplied documentation about
collection method, consent, time period, or representativeness. Results are
exploratory and cannot establish reliable real-world performance.
