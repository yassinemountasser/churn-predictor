import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. Load Data
df = pd.read_csv('customer_data.csv')

# 2. Separate Features (X) and Target (y)
X = df.drop(['CustomerID', 'Churn'], axis=1) # Drop ID (useless) and Target
y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0) # Convert Yes/No to 1/0

# 3. Define Preprocessing
# We need to turn text (Male/Female) into numbers (0/1) for the machine.
categorical_cols = ['Gender', 'Contract', 'InternetService']
numerical_cols = ['SeniorCitizen', 'Tenure', 'MonthlyCharges', 'TotalCharges']

# This transformer handles the conversion automatically
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_cols),
        ('num', 'passthrough', numerical_cols)
    ]
)

# 4. calculate scale_pos_weight
# This tells the model: "Pay 3x more attention to 'Yes' answers because they are rare."
# Formula: (Count of Negatives) / (Count of Positives)
pos_weight = sum(y==0) / sum(y==1)
print(f"Calculated Class Weighting: {pos_weight:.2f}")

# 5. Create the Full Pipeline
# Step 1: Preprocess Data -> Step 2: Train XGBoost Model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        random_state=42,
        scale_pos_weight=pos_weight, # Handles the imbalance!
        eval_metric='logloss'
    ))
])

# 6. Split Data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train
print("\nTraining Model...")
model.fit(X_train, y_train)

# 8. Evaluate
print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test)

# Confusion Matrix shows: [True Neg, False Pos]
#                         [False Neg, True Pos]
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 9. Save the Model
# We save the WHOLE pipeline, so we don't need to rewrite preprocessing code later.
joblib.dump(model, 'churn_model.pkl')
print("✅ Model saved as 'churn_model.pkl'")