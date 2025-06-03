
# Capstone Phase 5 Group 9
**Predicting Customer Churn Risk in Internet Service Subscribers using Machine Learning.**

## Project Structure
- `data/`: Contains the dataset used in the project.
- `notebooks/`: Jupyter notebooks for EDA, modeling, and evaluation.
- `reports/`: Project proposal and documentation.

## Dataset
A CSV dataset containing 36,992 customer records with 23 features.
## Key target Variable 
'churn' (Yes/No recorded after 30 days of inactivity or cancellation)

## Objectives
- Predict customer churn risk with ≥ 70 % recall (high‑risk segment)
- Uncover the drivers of churn at both global and segment levels
- Incorporate customer feedback & sentiment into predictive features
- Recommend actionable strategies to reduce churn by at least 10 % YoY

## Business Benefits
- Reduced Acquisition Cost - Retaining existing users is ~5× cheaper than 
  acquiring new ones
- Higher CLV - Targeted offers extend customer lifetime and loyalty
- Customer-Centric Innovation - Complaint and sentiment analysis fuels product/service improvements
- Revenue Stabilization - Early churn signals allow proactive interventions

# Methodolpgy
## 1. Business Understanding
- Clearly defined churn as customers with 30+ days of inactivity or confirmed cancellation.
- Identified key target KPIs: retention rate and customer lifetime value (CLV).
- Established success criteria: predictive model must achieve recall ≥ 70 % on churned customers.
## 2. Data Understanding
- Performed descriptive statistics to understand distribution and outliers
- Conducted missing-value analysis to identify and handle gaps in data
- Merged all datasets using the unified customer_id key for consistency and 
  integration
## 3. Data Preprocessing & Feature Engineering
- **Handling Missing Values**: < 1.7 % rows dropped (negligible).

- **Standardization**: Centered and scaled continuous variables.

- **Normalization**: Rescaled skewed features (e.g., avg_session_length).

- **Encoding**: One‑hot for nominal, ordinal for membership tiers.

- **New Features**: Complaint‑resolution latency, rolling 30‑day login 
   frequency, sentiment score via VADER.
## 4. Modeling
    | Model             | Key Hyperparameters              | ROC‑AUC | Recall | Notes                    |
|-------------------|----------------------------------|---------|--------|--------------------------|
| Logistic Regression | C, class_weight                 | 0.84    | 0.72   | Baseline                 |
| Random Forest     | n_estimators, max_depth          | 0.91    | 0.81   | Robust, interpretable    |
| XGBoost           | eta, max_depth, subsample        | 0.94    | 0.85   | Best overall             |
| LightGBM          | learning_rate, num_leaves        | 0.93    | 0.84   | Fast, low memory         |
| MLP (Keras)       | 3 hidden layers, dropout          | 0.92    | 0.83   | Captures non‑linearities |

## 5. Evaluation & Interpretation

- **Confusion Matrix**: Assessed the number of true/false positives and negatives to evaluate model accuracy.
- **ROC‑AUC**: Measured the model’s ability to distinguish between churned and retained customers.
- **Precision‑Recall Curves**: Evaluated performance on imbalanced data, focusing on the trade‑off between precision and recall.
- **SHAP Values**: Identified and visualized top features driving churn predictions.

## Exploratory Data Analysis Highlights

- Churn rate peaks at 27 % among Basic members vs. 8 % for Gold.
- Complaints double churn likelihood—even when resolved within SLA.
- Wallet engagement (points earned/redeemed) inversely correlates with churn (Pearson = ‑0.47).
- Urban customers churn 1.6× more than suburban equivalents, likely due to fierce competition.

## Key Insights & Recommendations

- **Upgrade Incentives** - Offer tiered discounts to nudge Basic members to Silver within first 90 days.
- **Complaint Concierge** - Introduce a “white‑glove” resolution team targeting high‑value accounts.
- **Dynamic Loyalty Points** - Gamify wallet usage with streak bonuses and badges.
- **Geo‑Targeted Retention** - Prioritize retention ads in high‑churn metro postcodes.
- **Early‑Warning Triggers** - Automate offers when churn‑probability > 0.65 and loyalty engagement < 10 pts/mo.


   


