"""
Comprehensive Test Data Generator for Explainable AI Classification Application

This script generates various synthetic datasets and trains different classification models
to test the capabilities of the Classification Stats application.

Features:
- Multiple dataset types with different characteristics
- Various classification algorithms
- Realistic data with different complexity levels
- Separate train/test datasets for proper evaluation
- Models with different performance characteristics (good, overfitted, underfitted)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_blobs, make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import joblib
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TestDataGenerator:
    def __init__(self, output_dir="test_datasets"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_dataset_1_high_dimensional_balanced(self):
        """
        High-dimensional balanced dataset with informative features
        Good for testing feature importance and SHAP explanations
        """
        print("Generating Dataset 1: High-dimensional balanced dataset...")
        
        X, y = make_classification(
            n_samples=2000,
            n_features=20,
            n_informative=15,
            n_redundant=3,
            n_clusters_per_class=2,
            class_sep=1.2,
            random_state=42
        )
        
        # Create meaningful feature names
        feature_names = [
            'income_score', 'credit_history', 'employment_years', 'debt_ratio', 
            'savings_amount', 'education_level', 'age_group', 'property_value',
            'monthly_expenses', 'investment_portfolio', 'insurance_coverage',
            'loan_history', 'payment_behavior', 'financial_stability',
            'risk_tolerance', 'market_volatility', 'economic_indicator',
            'demographic_factor', 'lifestyle_score', 'social_score'
        ]
        
        df = pd.DataFrame(X, columns=feature_names)
        df['approval_status'] = y
        
        return df, "Financial Approval Prediction"
    
    def generate_dataset_2_imbalanced_medical(self):
        """
        Imbalanced medical dataset - good for testing precision/recall tradeoffs
        """
        print("Generating Dataset 2: Imbalanced medical dataset...")
        
        # Create imbalanced dataset
        X, y = make_classification(
            n_samples=1500,
            n_features=15,
            n_informative=12,
            n_redundant=2,
            n_clusters_per_class=1,
            weights=[0.85, 0.15],  # 85% negative, 15% positive
            class_sep=0.8,
            random_state=123
        )
        
        feature_names = [
            'blood_pressure', 'cholesterol_level', 'glucose_level', 'bmi_score',
            'heart_rate', 'age', 'family_history', 'smoking_status',
            'exercise_frequency', 'stress_level', 'sleep_quality',
            'medication_count', 'symptoms_severity', 'lab_result_1', 'lab_result_2'
        ]
        
        df = pd.DataFrame(X, columns=feature_names)
        df['disease_risk'] = y
        
        return df, "Medical Disease Risk Prediction"
    
    def generate_dataset_3_nonlinear_complex(self):
        """
        Complex non-linear dataset with interactions
        """
        print("Generating Dataset 3: Complex non-linear dataset...")
        
        # Generate base features
        n_samples = 1800
        X1 = np.random.normal(0, 1, (n_samples, 5))
        X2 = np.random.normal(0, 1, (n_samples, 5))
        
        # Create complex interactions
        feature_interactions = np.column_stack([
            X1[:, 0] * X2[:, 1],  # interaction 1
            X1[:, 2] ** 2,        # quadratic
            np.sin(X1[:, 3]),     # trigonometric
            np.exp(X2[:, 0] * 0.5), # exponential
            X1[:, 4] * X2[:, 2] * X2[:, 3]  # three-way interaction
        ])
        
        X = np.column_stack([X1, X2, feature_interactions])
        
        # Create complex decision boundary
        y = (X[:, 0] + X[:, 5] + X[:, 10] + 
             0.5 * X[:, 12] - 0.3 * X[:, 13] + 
             np.random.normal(0, 0.1, n_samples)) > 0.5
        y = y.astype(int)
        
        feature_names = [
            'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
            'temperature', 'humidity', 'pressure', 'vibration', 'noise_level',
            'interaction_temp_humid', 'quadratic_sensor', 'sine_transform',
            'exp_temperature', 'complex_interaction'
        ]
        
        df = pd.DataFrame(X, columns=feature_names)
        df['equipment_failure'] = y
        
        return df, "Industrial Equipment Failure Prediction"
    
    def generate_dataset_4_low_signal_noise(self):
        """
        Low signal-to-noise ratio dataset - challenging for models
        """
        print("Generating Dataset 4: Low signal-to-noise dataset...")
        
        X, y = make_classification(
            n_samples=1200,
            n_features=25,
            n_informative=5,  # Only 5 out of 25 features are informative
            n_redundant=5,
            n_clusters_per_class=1,
            class_sep=0.6,  # Low class separation
            random_state=789
        )
        
        # Add extra noise
        noise = np.random.normal(0, 0.5, X.shape)
        X = X + noise
        
        feature_names = [f'feature_{i+1}' for i in range(25)]
        
        df = pd.DataFrame(X, columns=feature_names)
        df['target_class'] = y
        
        return df, "Noisy Feature Classification"
    
    def generate_dataset_5_categorical_mixed(self):
        """
        Mixed categorical and numerical features
        """
        print("Generating Dataset 5: Mixed categorical-numerical dataset...")
        
        n_samples = 1600
        
        # Numerical features
        numerical_features = np.random.normal(0, 1, (n_samples, 8))
        
        # Categorical features
        categories_1 = np.random.choice(['A', 'B', 'C', 'D'], n_samples)
        categories_2 = np.random.choice(['Type1', 'Type2', 'Type3'], n_samples)
        categories_3 = np.random.choice(['Low', 'Medium', 'High'], n_samples)
        
        # Create target based on mixed features
        target_prob = (
            numerical_features[:, 0] + 
            numerical_features[:, 3] * 0.8 +
            (categories_1 == 'A').astype(float) * 0.5 +
            (categories_2 == 'Type3').astype(float) * 0.7 +
            (categories_3 == 'High').astype(float) * 0.6 +
            np.random.normal(0, 0.2, n_samples)
        )
        
        y = (target_prob > np.median(target_prob)).astype(int)
        
        # Create DataFrame
        df = pd.DataFrame({
            'numerical_1': numerical_features[:, 0],
            'numerical_2': numerical_features[:, 1],
            'numerical_3': numerical_features[:, 2],
            'numerical_4': numerical_features[:, 3],
            'numerical_5': numerical_features[:, 4],
            'numerical_6': numerical_features[:, 5],
            'numerical_7': numerical_features[:, 6],
            'numerical_8': numerical_features[:, 7],
            'category_main': categories_1,
            'category_type': categories_2,
            'category_level': categories_3
        })
        
        # Encode categorical variables
        le1 = LabelEncoder()
        le2 = LabelEncoder()
        le3 = LabelEncoder()
        
        df['category_main_encoded'] = le1.fit_transform(df['category_main'])
        df['category_type_encoded'] = le2.fit_transform(df['category_type'])
        df['category_level_encoded'] = le3.fit_transform(df['category_level'])
        
        # Drop original categorical columns for model training
        df_encoded = df.drop(['category_main', 'category_type', 'category_level'], axis=1)
        df_encoded['customer_segment'] = y
        
        return df_encoded, "Customer Segmentation Prediction"
    
    def train_models_variety(self, X_train, X_test, y_train, y_test, dataset_name):
        """
        Train multiple types of models with different characteristics
        """
        models_config = {
            'random_forest_balanced': {
                'model': RandomForestClassifier(
                    n_estimators=100, 
                    max_depth=10, 
                    min_samples_split=5,
                    random_state=42
                ),
                'description': 'Well-balanced Random Forest'
            },
            'random_forest_overfitted': {
                'model': RandomForestClassifier(
                    n_estimators=200, 
                    max_depth=None,  # No depth limit
                    min_samples_split=2,  # Minimum splits
                    random_state=42
                ),
                'description': 'Potentially overfitted Random Forest'
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                ),
                'description': 'Gradient Boosting Classifier'
            },
            'logistic_regression': {
                'model': LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    random_state=42
                ),
                'description': 'Logistic Regression'
            },
            'decision_tree_simple': {
                'model': DecisionTreeClassifier(
                    max_depth=5,
                    min_samples_split=10,
                    random_state=42
                ),
                'description': 'Simple Decision Tree'
            },
            'extra_trees': {
                'model': ExtraTreesClassifier(
                    n_estimators=100,
                    max_depth=12,
                    random_state=42
                ),
                'description': 'Extra Trees Classifier'
            }
        }
        
        trained_models = {}
        
        for model_name, config in models_config.items():
            print(f"  Training {config['description']}...")
            
            model = config['model']
            
            # Scale features for models that benefit from it
            if model_name in ['logistic_regression']:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                model.fit(X_train_scaled, y_train)
                
                # Save scaler along with model info
                trained_models[model_name] = {
                    'model': model,
                    'scaler': scaler,
                    'X_train': X_train_scaled,
                    'X_test': X_test_scaled,
                    'description': config['description']
                }
            else:
                model.fit(X_train, y_train)
                trained_models[model_name] = {
                    'model': model,
                    'scaler': None,
                    'X_train': X_train,
                    'X_test': X_test,
                    'description': config['description']
                }
        
        return trained_models
    
    def save_datasets_and_models(self, df, dataset_name, dataset_description):
        """
        Save datasets and train multiple models
        """
        # Prepare features and target
        target_col = df.columns[-1]  # Last column is target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Create train and test DataFrames
        train_df = X_train.copy()
        train_df[target_col] = y_train
        
        test_df = X_test.copy()
        test_df[target_col] = y_test
        
        # Save datasets
        dataset_dir = os.path.join(self.output_dir, dataset_name.lower().replace(' ', '_'))
        os.makedirs(dataset_dir, exist_ok=True)
        
        train_df.to_csv(os.path.join(dataset_dir, 'train_dataset.csv'), index=False)
        test_df.to_csv(os.path.join(dataset_dir, 'test_dataset.csv'), index=False)
        
        # Train multiple models
        trained_models = self.train_models_variety(X_train, X_test, y_train, y_test, dataset_name)
        
        # Save models and create summary
        model_summary = {
            'dataset_name': dataset_name,
            'dataset_description': dataset_description,
            'target_column': target_col,
            'n_samples_train': len(train_df),
            'n_samples_test': len(test_df),
            'n_features': len(X.columns),
            'feature_names': list(X.columns),
            'class_distribution_train': y_train.value_counts().to_dict(),
            'class_distribution_test': y_test.value_counts().to_dict(),
            'models': {}
        }
        
        for model_name, model_info in trained_models.items():
            # Save model
            model_path = os.path.join(dataset_dir, f'{model_name}_model.joblib')
            joblib.dump(model_info['model'], model_path)
            
            # Save scaler if exists
            if model_info['scaler'] is not None:
                scaler_path = os.path.join(dataset_dir, f'{model_name}_scaler.joblib')
                joblib.dump(model_info['scaler'], scaler_path)
            
            # Calculate basic performance metrics
            if model_info['scaler'] is not None:
                train_score = model_info['model'].score(model_info['X_train'], y_train)
                test_score = model_info['model'].score(model_info['X_test'], y_test)
            else:
                train_score = model_info['model'].score(X_train, y_train)
                test_score = model_info['model'].score(X_test, y_test)
            
            model_summary['models'][model_name] = {
                'description': model_info['description'],
                'model_path': model_path,
                'scaler_path': scaler_path if model_info['scaler'] is not None else None,
                'train_accuracy': round(train_score, 4),
                'test_accuracy': round(test_score, 4),
                'overfitting_gap': round(train_score - test_score, 4)
            }
        
        # Save summary
        import json
        with open(os.path.join(dataset_dir, 'summary.json'), 'w') as f:
            json.dump(model_summary, f, indent=2)
        
        print(f"✓ Saved {dataset_name} with {len(trained_models)} models in {dataset_dir}")
        return model_summary
    
    def generate_all_test_data(self):
        """
        Generate all test datasets and models
        """
        print("=" * 60)
        print("GENERATING COMPREHENSIVE TEST DATA FOR CLASSIFICATION STATS")
        print("=" * 60)
        
        datasets = [
            self.generate_dataset_1_high_dimensional_balanced(),
            self.generate_dataset_2_imbalanced_medical(),
            self.generate_dataset_3_nonlinear_complex(),
            self.generate_dataset_4_low_signal_noise(),
            self.generate_dataset_5_categorical_mixed()
        ]
        
        all_summaries = []
        
        for df, description in datasets:
            print(f"\nProcessing: {description}")
            print("-" * 40)
            summary = self.save_datasets_and_models(df, description, description)
            all_summaries.append(summary)
        
        # Create master summary
        master_summary = {
            'generation_date': datetime.now().isoformat(),
            'total_datasets': len(all_summaries),
            'total_models': sum(len(s['models']) for s in all_summaries),
            'datasets': all_summaries
        }
        
        with open(os.path.join(self.output_dir, 'master_summary.json'), 'w') as f:
            json.dump(master_summary, f, indent=2)
        
        print("\n" + "=" * 60)
        print("GENERATION COMPLETE!")
        print("=" * 60)
        print(f"Generated {len(all_summaries)} datasets with multiple models each")
        print(f"Total models created: {sum(len(s['models']) for s in all_summaries)}")
        print(f"Output directory: {os.path.abspath(self.output_dir)}")
        
        # Print testing instructions
        self.print_testing_instructions()
        
        return master_summary
    
    def print_testing_instructions(self):
        """
        Print instructions for testing the application
        """
        print("\n" + "=" * 60)
        print("TESTING INSTRUCTIONS")
        print("=" * 60)
        
        print("\n1. DATASET VARIETY:")
        print("   • High-dimensional balanced: Test feature importance rankings")
        print("   • Imbalanced medical: Test precision/recall tradeoffs and thresholds")
        print("   • Non-linear complex: Test model's ability to capture interactions")
        print("   • Low signal-noise: Test robustness with noisy data")
        print("   • Mixed categorical: Test handling of different feature types")
        
        print("\n2. MODEL VARIETY:")
        print("   • Balanced Random Forest: Good baseline performance")
        print("   • Overfitted Random Forest: Test overfitting detection")
        print("   • Gradient Boosting: Different algorithm characteristics")
        print("   • Logistic Regression: Linear model comparison")
        print("   • Decision Tree: Simple interpretable model")
        print("   • Extra Trees: Ensemble method variation")
        
        print("\n3. TESTING SCENARIOS:")
        print("   • Upload different model types to see varied SHAP explanations")
        print("   • Compare overfitted vs balanced models using train/test datasets")
        print("   • Test threshold analysis with imbalanced datasets")
        print("   • Verify ROC curves and AUC calculations")
        print("   • Test AI explanation generation with different model types")
        
        print("\n4. FILES TO USE:")
        print("   • Each dataset folder contains:")
        print("     - train_dataset.csv (for training data)")
        print("     - test_dataset.csv (for test data)")
        print("     - *_model.joblib (trained models)")
        print("     - *_scaler.joblib (feature scalers, if needed)")
        print("     - summary.json (detailed information)")
        
        print(f"\n5. START TESTING:")
        print(f"   • Navigate to: {os.path.abspath(self.output_dir)}")
        print("   • Choose any dataset folder")
        print("   • Upload the model.joblib file and corresponding train/test CSV files")
        print("   • Explore the Classification Stats page features!")

def main():
    """
    Main function to generate all test data
    """
    generator = TestDataGenerator()
    summary = generator.generate_all_test_data()
    
    print(f"\n📊 Test data generation completed successfully!")
    print(f"📁 Check the '{generator.output_dir}' folder for all generated files")
    
    return summary

if __name__ == "__main__":
    main()
