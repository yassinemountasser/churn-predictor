# 📉 Customer Retention Engine (Churn Predictor)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Library](https://img.shields.io/badge/Model-XGBoost-orange)
![Framework](https://img.shields.io/badge/Dashboard-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Complete-success)

## 📖 Project Overview
This project is an end-to-end **Machine Learning pipeline** designed to predict customer churn for subscription-based businesses.

In a SaaS or Telco environment, retaining an existing customer is significantly cheaper than acquiring a new one. This tool helps stakeholders identify "at-risk" customers *before* they leave, enabling proactive retention strategies.

**Key Features:**
* **Production-Ready Pipeline:** Uses `Scikit-Learn` pipelines to bundle preprocessing and model logic, preventing training-serving skew.
* **Imbalanced Data Handling:** Implements `scale_pos_weight` to correctly penalize false negatives, addressing the common issue where "No Churn" classes dominate the dataset.
* **Interactive Dashboard:** A `Streamlit` interface that allows non-technical users to simulate scenarios (e.g., "If we offer this user a 1-year contract, does their risk drop?").

## 🛠️ Technical Architecture

### The Stack
* **Core:** Python
* **Machine Learning:** XGBoost (Extreme Gradient Boosting)
* **Data Processing:** Pandas, NumPy, Scikit-Learn
* **Visualization:** Plotly, Streamlit

### The Workflow
1.  **Data Ingestion:** Synthetic data generation mimicking real-world Telco attributes (Tenure, Contract Type, Monthly Charges).
2.  **Preprocessing:** * One-Hot Encoding for categorical variables (Gender, Internet Service).
    * Automated feature scaling via `ColumnTransformer`.
3.  **Modeling:** * Trained on an XGBoost Classifier optimized for recall.
    * Probability calibration to output specific risk percentages rather than binary Yes/No predictions.
4.  **Deployment:** Model serialized via `joblib` and served through a local Streamlit web app.

## 📊 Performance & Insights
The model prioritizes **Recall** (Sensitivity) over Precision. 
* *Business Logic:* Missing a churning customer (False Negative) is costly, whereas offering a discount to a loyal customer (False Positive) is an acceptable retention cost.
* *Sanity Checks:* The model successfully identifies high-risk archetypes (e.g., Month-to-month contracts with high monthly bills) vs. low-risk archetypes (Long-term tenure).

How to Run Locally:

1. Clone the repository
```bash
git clone [https://github.com/yassinemountasser/churn-predictor.git](https://github.com/yassinemountasser/churn-predictor.git)
cd churn-predictor
```
2. Install Dependencies
```bash
pip install -r requirements.txt
```
3. Generate Data & Train Model
```bash
python generate_data.py
python train_model.py
```
4. Launch the Dashboard
```bash
streamlit run app.py
```
