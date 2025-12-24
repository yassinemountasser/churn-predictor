import pandas as pd
import plotly.express as px

# 1. Load Data
df = pd.read_csv('customer_data.csv')

# 2. Quick Check
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# 3. Check for Class Imbalance (Crucial for Churn projects!)
churn_counts = df['Churn'].value_counts()
print("\nChurn Distribution:")
print(churn_counts)

# 4. Create a quick visualization
fig = px.histogram(df, x='Churn', color='Churn', title="Churn Distribution Check")
fig.show()

# 5. Check Correlations (Numeric only)
# We need to map 'Yes'/'No' to 1/0 to see correlations
df['Churn_Numeric'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# See how Numeric features correlate with Churn
print("\nCorrelations with Churn:")
print(df.select_dtypes(include=['number']).corr()['Churn_Numeric'].sort_values(ascending=False))