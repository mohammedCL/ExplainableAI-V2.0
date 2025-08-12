import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Set random seed for reproducibility
np.random.seed(42)

# Generate a synthetic dataset with meaningful feature interactions
n_samples = 1000

# Create correlated features that will show interesting interactions
age = np.random.normal(35, 12, n_samples)
age = np.clip(age, 18, 70)  # Clip to realistic age range

income = age * 1200 + np.random.normal(0, 8000, n_samples)  # Income correlates with age
income = np.clip(income, 20000, 120000)  # Realistic income range

education_years = np.random.normal(14, 3, n_samples)
education_years = np.clip(education_years, 8, 20)

# Credit score influenced by age and income
credit_score = (age * 8 + income * 0.002 + education_years * 15 + 
                np.random.normal(0, 50, n_samples))
credit_score = np.clip(credit_score, 300, 850)

# Debt-to-income ratio (inversely related to income)
debt_ratio = np.maximum(0.1, 0.8 - (income - 30000) / 100000 + 
                       np.random.normal(0, 0.15, n_samples))
debt_ratio = np.clip(debt_ratio, 0.05, 0.95)

# Employment length (related to age)
employment_length = np.maximum(0, age - 22 + np.random.normal(0, 5, n_samples))
employment_length = np.clip(employment_length, 0, 40)

# Savings (related to income and age)
savings = (income * 0.1 + age * 500 + np.random.normal(0, 5000, n_samples))
savings = np.maximum(0, savings)

# Create target variable: loan approval (influenced by multiple factors)
# Higher chance of approval with: higher credit score, lower debt ratio, 
# higher income, more savings
approval_score = (credit_score * 0.8 + 
                 (1 - debt_ratio) * 200 + 
                 income * 0.001 + 
                 savings * 0.0001 + 
                 employment_length * 2 +
                 np.random.normal(0, 50, n_samples))

# Convert to binary classification (approved/rejected)
loan_approved = (approval_score > np.percentile(approval_score, 40)).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'age': age.round(0),
    'income': income.round(0),
    'education_years': education_years.round(1),
    'credit_score': credit_score.round(0),
    'debt_to_income_ratio': debt_ratio.round(3),
    'employment_length': employment_length.round(1),
    'savings': savings.round(0),
    'loan_approved': loan_approved
})

# Add some categorical features for more interesting interactions
df['income_category'] = pd.cut(df['income'], 
                              bins=[0, 40000, 70000, float('inf')], 
                              labels=['Low', 'Medium', 'High'])

df['age_group'] = pd.cut(df['age'], 
                        bins=[0, 30, 50, float('inf')], 
                        labels=['Young', 'Middle', 'Senior'])

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=['income_category', 'age_group'], prefix=['income', 'age'])

# Save the dataset
dataset_filename = 'loan_approval_dataset.csv'
df_encoded.to_csv(dataset_filename, index=False)
print(f"✓ Saved '{dataset_filename}' in {os.getcwd()}")

# Prepare data for model training
X = df_encoded.drop(columns=['loan_approved'])
y = df_encoded['loan_approved']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a Random Forest model (good for feature importance analysis)
# Note: Random Forest doesn't need feature scaling
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_train, y_train)

# Evaluate the model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Model Performance:")
print(f"  Training Accuracy: {train_score:.4f}")
print(f"  Test Accuracy: {test_score:.4f}")

# Save the trained model only
model_filename = 'loan_approval_model.joblib'
joblib.dump(model, model_filename)
print(f"✓ Saved '{model_filename}' in {os.getcwd()}")

# Display feature importance for interaction analysis
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop Feature Importances:")
for idx, row in feature_importance.head(8).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Display dataset information
print(f"\nDataset Information:")
print(f"  Dataset shape: {df_encoded.shape}")
print(f"  Features: {len(X.columns)}")
print(f"  Numerical features: age, income, education_years, credit_score, debt_to_income_ratio, employment_length, savings")
print(f"  Categorical features: income_category, age_group (one-hot encoded)")
print(f"  Target: loan_approved (Binary: 0=Rejected, 1=Approved)")
print(f"  Class distribution:")
print(f"    Rejected (0): {sum(y == 0)} samples ({sum(y == 0)/len(y)*100:.1f}%)")
print(f"    Approved (1): {sum(y == 1)} samples ({sum(y == 1)/len(y)*100:.1f}%)")

print(f"\nExpected Feature Interactions:")
print("  • Age ↔ Income (positive correlation)")
print("  • Income ↔ Credit Score (positive correlation)")  
print("  • Debt Ratio ↔ Income (negative correlation)")
print("  • Employment Length ↔ Age (positive correlation)")
print("  • Savings ↔ Income + Age (positive correlation)")
print("  • Multiple features → Loan Approval")

print(f"\nFiles saved successfully in: {os.getcwd()}")
print(f"  1. {dataset_filename} - Complete dataset with features and target")
print(f"  2. {model_filename} - Trained RandomForest classifier")

print(f"\n" + "="*50)
print(f"FOR YOUR APPLICATION:")
print(f"Target Column Name: 'loan_approved'")
print(f"Model File: {model_filename}")
print(f"Dataset File: {dataset_filename}")
print(f"="*50)