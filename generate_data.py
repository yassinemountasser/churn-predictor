import pandas as pd
import numpy as np
import random

def generate_dummy_data(n_rows=1000):
    print(f"Generating {n_rows} rows of fake customer data...")
    
    # 1. Setup random seed for reproducibility
    np.random.seed(42)
    
    # 2. Define categories
    ids = [f"CUST-{i:04d}" for i in range(n_rows)]
    genders = ['Male', 'Female']
    contracts = ['Month-to-Month', 'One Year', 'Two Year']
    internet_types = ['DSL', 'Fiber Optic', 'No Internet']
    
    data = []
    
    for i in range(n_rows):
        cid = ids[i]
        gender = random.choice(genders)
        senior = random.choice([0, 1]) # 0 = No, 1 = Yes
        contract = random.choice(contracts)
        internet = random.choice(internet_types)
        
        # 3. Create correlations (Make the data "make sense")
        # People with longer tenure usually pay more over time but might have lower monthly costs
        tenure_months = np.random.randint(1, 73)
        
        base_price = 30
        if internet == 'Fiber Optic': base_price += 40
        if internet == 'DSL': base_price += 20
        
        # Add some random variance to the bill
        monthly_charges = base_price + np.random.uniform(-5, 15)
        
        # Calculate total charges (Tenure * Monthly)
        total_charges = tenure_months * monthly_charges
        
        # 4. Simulate Churn Logic (The Target Variable)
        # "Month-to-Month" contracts are more likely to churn
        churn_prob = 0.1
        if contract == 'Month-to-Month': churn_prob += 0.3
        if internet == 'Fiber Optic': churn_prob += 0.1 # Expensive service = higher churn risk
        if tenure_months < 12: churn_prob += 0.1 # New customers leave more often
        
        # Cap probability at 0.9 and determine churn
        churn_prob = min(churn_prob, 0.9)
        churn = 'Yes' if random.random() < churn_prob else 'No'
        
        data.append([cid, gender, senior, tenure_months, contract, internet, 
                     round(monthly_charges, 2), round(total_charges, 2), churn])
    
    # 5. Create DataFrame
    columns = ['CustomerID', 'Gender', 'SeniorCitizen', 'Tenure', 'Contract', 
               'InternetService', 'MonthlyCharges', 'TotalCharges', 'Churn']
    
    df = pd.DataFrame(data, columns=columns)
    
    # 6. Save to CSV
    df.to_csv('customer_data.csv', index=False)
    print("✅ Success! 'customer_data.csv' has been created.")
    return df

if __name__ == "__main__":
    generate_dummy_data()