# Customer Churn Prediction & Retention Strategy

**Business problem:** A telecom company is losing customers to churn but doesn't know
who's at risk or where to focus retention spend. This project builds an end-to-end
pipeline — from SQL-based exploratory analysis to a deployed predictive model — that
identifies at-risk customers and quantifies the ROI of acting on them.

**Live demo:** [add your deployed Streamlit link here]

## Key findings
- Overall churn rate: **30.7%** (5,000 customers analyzed)
- Month-to-month contracts churn at **37.7%** vs **20.9%** for two-year contracts — nearly 2x
- New customers (0–6 months tenure) churn at **41.6%** — onboarding is a clear weak point
- Highest-risk segment (month-to-month + no tech support + pays by electronic check):
  **44.7% churn rate**, ~$65 avg monthly charge
- **$841K/year** in revenue is at risk from month-to-month customers alone

## Model & business impact
- Trained Logistic Regression (interpretable baseline) and Random Forest classifiers
- Chose to optimize for **recall over raw accuracy** — a missed churner costs more than a
  wasted retention offer, so catching more true churners matters more than fewer false alarms
- Targeting the top 20% highest-risk customers captures **69.6% of actual churners**
- Simulated retention campaign ($15/customer outreach, 30% assumed save rate):
  **10.9x ROI** ($15K spend → ~$179K revenue saved)

## Tech stack
- **SQL** (sqlite3) for business-question-driven EDA
- **Python** (pandas, scikit-learn) for feature engineering and modeling
- **Streamlit + Plotly** for the interactive dashboard
- Logistic Regression + Random Forest, evaluated on precision/recall/F1/ROC-AUC with
  stratified train/test split to preserve class balance

## Repo structure
```
generate_data.py      # synthetic dataset generation (realistic churn drivers)
sql_eda.py             # SQL business-question analysis
model.py                # training + evaluation + feature importance
business_impact.py      # ROI calculation from model scores
streamlit_app.py         # deployable interactive dashboard
telco_churn.csv           # dataset
scored_customers.csv       # every customer with a churn risk score
churn_model.pkl              # trained model artifact
```

## Run locally
```bash
pip install pandas numpy scikit-learn streamlit plotly joblib
python generate_data.py
python model.py
python business_impact.py
streamlit run streamlit_app.py
```

## Deploy free
Push this repo to GitHub, go to [streamlit.io/cloud](https://streamlit.io/cloud),
connect your repo, and point it at `streamlit_app.py`. You'll get a live public URL
in about 2 minutes — put that link on your resume next to this project.

## What I'd do with more data
- Real customer data (this uses a synthetic dataset built to mirror realistic churn
  drivers, since it was built without live data access)
- A/B test the retention campaign itself rather than assuming a 30% save rate
- Add SHAP values for per-customer explanation ("why is this customer high-risk")
