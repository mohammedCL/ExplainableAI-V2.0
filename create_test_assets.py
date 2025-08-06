import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Save the full dataset for upload
df.to_csv('breast_cancer_dataset.csv', index=False)
print("Saved 'breast_cancer_dataset.csv'")

# Prepare data for model training
X = df.drop(columns=['target'])
y = df['target']
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Save the model file
joblib.dump(model, 'cancer_model.joblib')
print("Saved 'cancer_model.joblib'")