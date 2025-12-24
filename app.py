import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# 1. Load the trained model pipeline
# This loads the logic + the preprocessing steps we saved earlier.
model = joblib.load('churn_model.pkl')

# 2. Page Configuration
st.set_page_config(page_title="Churn Predictor", layout="wide")

st.title("📊 Customer Retention Engine")
st.markdown("""
This tool uses an **XGBoost Machine Learning model** to predict if a customer is at risk of cancelling their service (Churn).
Adjust the customer profile on the left to see how different factors impact retention.
""")

# 3. Sidebar for User Inputs
st.sidebar.header("Customer Profile")

def user_input_features():
    # We need to capture the same features we trained on:
    # Gender, SeniorCitizen, Tenure, Contract, InternetService, MonthlyCharges, TotalCharges
    
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    senior = st.sidebar.selectbox("Senior Citizen?", (0, 1), format_func=lambda x: "Yes" if x == 1 else "No")
    contract = st.sidebar.selectbox("Contract Type", ("Month-to-Month", "One Year", "Two Year"))
    internet = st.sidebar.selectbox("Internet Service", ("DSL", "Fiber Optic", "No Internet"))
    
    tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 12)
    monthly_charges = st.sidebar.slider("Monthly Charges ($)", 20.0, 120.0, 70.0)
    
    # In our data, TotalCharges was roughly Tenure * Monthly. 
    # Let's auto-calculate it for the user to keep it simple.
    total_charges = tenure * monthly_charges
    
    data = {
        'Gender': gender,
        'SeniorCitizen': senior,
        'Contract': contract,
        'InternetService': internet,
        'Tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Display the User's selection
st.subheader("Current Customer Profile")
st.dataframe(input_df)

# 4. Make Prediction
if st.button("Analyze Risk"):
    
    # The pipeline automatically handles the text conversion (OneHotEncoding)
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    
    # probability returns [Prob of No, Prob of Yes]
    churn_risk = probability[0][1] 
    
    # 5. Visualizing the Result
    st.write("---")
    st.subheader("Prediction Result")
    
    # Create a Gauge Chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = churn_risk * 100,
        title = {'text': "Churn Probability (%)"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps' : [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}],
            'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}
        }
    ))
    
    st.plotly_chart(fig)
    
    # Text interpretation
    if churn_risk > 0.5:
        st.error(f"⚠️ High Risk! This customer has a {churn_risk:.1%} chance of leaving.")
        st.info("💡 Recommendation: Consider offering a 1-year contract discount.")
    else:
        # We calculate the probability of staying (1 - risk)
        # We let the f-string handles the percentage conversion automatically
        stay_prob = 1 - churn_risk
        st.success(f"✅ Low Risk. This customer is likely to stay ({stay_prob:.1%} confidence).")