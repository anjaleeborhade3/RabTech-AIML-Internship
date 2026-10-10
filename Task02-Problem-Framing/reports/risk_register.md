# Risk Register — Customer Churn Prediction

| ID | Risk | Likelihood | Impact | Mitigation | Monitoring / rollback |
|---|---|---|---|---|---|
| R1 | Dataset too small (12 rows) | High | High | Educational use only; collect a larger representative dataset | Do not deploy until validation is adequate |
| R2 | Unknown consent, license, and provenance | Unknown | High | Ask the provider to confirm source, permission, and usage terms | Do not redistribute or use commercially until verified |
| R3 | Target or feature timing may cause leakage | Medium | High | Define prediction timestamp and verify when every feature is available | Stop evaluation if post-outcome information is found |
| R4 | False positive causes unwanted retention contact | Medium | Medium | Make outreach optional and require human review | Monitor complaints; suspend outreach if harm appears |
| R5 | False negative misses a customer needing help | Medium | Medium | Keep normal customer-support channels available | Review missed cases; never rely only on the model |
| R6 | Bias or proxy effects | Unknown | High | Obtain suitable, lawfully permitted evaluation data before deployment | Suspend use if harmful disparities are identified |
| R7 | Data errors or distribution drift | Medium | High | Validate inputs and monitor data quality | Roll back to human-led processes if checks fail |
| R8 | Predictions used to penalize customers | Low/Unknown | High | Restrict use to supportive service improvement | Suspend access if use exceeds the approved purpose |

## Ownership

- Dataset provenance and permission: project lead / dataset provider
- Data quality and leakage checks: data/ML team
- Customer contact decisions: customer-support lead
- Monitoring and rollback: designated model owner

## Release rule

This starter dataset is for educational exploration only. Do not deploy the model or make consequential customer decisions from these results. Unknown risks must be investigated rather than treated as absent.
