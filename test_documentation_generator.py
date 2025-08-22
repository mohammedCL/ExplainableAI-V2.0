"""
Test Documentation Generator for Explainable AI Classification Application

This script generates comprehensive test documentation including:
- Test cases for each dataset and model combination
- Expected outcomes and validation criteria
- Performance benchmarks and analysis points
- Feature analysis expectations
- UI/UX validation checklist
"""

import pandas as pd
import json
import os
from datetime import datetime
import numpy as np

class TestDocumentationGenerator:
    def __init__(self, test_datasets_dir="test_datasets", output_dir="test_documentation"):
        self.test_datasets_dir = test_datasets_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def load_dataset_summaries(self):
        """Load all dataset summaries from the test_datasets directory"""
        summaries = []
        
        # Load master summary if it exists
        master_path = os.path.join(self.test_datasets_dir, 'master_summary.json')
        if os.path.exists(master_path):
            try:
                with open(master_path, 'r') as f:
                    content = f.read().strip()
                    if content:  # Check if file is not empty
                        master_summary = json.loads(content)
                        return master_summary.get('datasets', [])
                    else:
                        print("Warning: master_summary.json is empty. Loading individual summaries...")
            except json.JSONDecodeError:
                print("Warning: master_summary.json is corrupted. Loading individual summaries...")
        
        # Otherwise, load individual summaries
        for folder in os.listdir(self.test_datasets_dir):
            folder_path = os.path.join(self.test_datasets_dir, folder)
            if os.path.isdir(folder_path):
                summary_path = os.path.join(folder_path, 'summary.json')
                if os.path.exists(summary_path):
                    try:
                        with open(summary_path, 'r') as f:
                            summaries.append(json.load(f))
                            print(f"Loaded: {folder}")
                    except json.JSONDecodeError:
                        print(f"Warning: Could not load {summary_path}")
        
        return summaries
    
    def generate_test_cases(self):
        """Generate comprehensive test cases"""
        summaries = self.load_dataset_summaries()
        test_cases = []
        
        for dataset in summaries:
            dataset_name = dataset['dataset_name']
            target_column = dataset['target_column']
            n_features = dataset['n_features']
            class_dist_train = dataset['class_distribution_train']
            class_dist_test = dataset['class_distribution_test']
            
            # Calculate imbalance ratio
            class_counts = list(class_dist_train.values())
            imbalance_ratio = max(class_counts) / min(class_counts) if len(class_counts) > 1 else 1.0
            
            for model_name, model_info in dataset['models'].items():
                test_case = {
                    # Basic Information
                    'test_id': f"{dataset_name.lower().replace(' ', '_')}_{model_name}",
                    'dataset_name': dataset_name,
                    'model_name': model_name,
                    'model_description': model_info['description'],
                    'target_column': target_column,
                    'n_features': n_features,
                    'n_samples_train': dataset['n_samples_train'],
                    'n_samples_test': dataset['n_samples_test'],
                    
                    # Dataset Characteristics
                    'dataset_type': self._categorize_dataset(dataset_name),
                    'imbalance_ratio': round(imbalance_ratio, 2),
                    'complexity_level': self._assess_complexity(dataset_name, n_features),
                    
                    # Model Performance
                    'train_accuracy': model_info['train_accuracy'],
                    'test_accuracy': model_info['test_accuracy'],
                    'overfitting_gap': model_info['overfitting_gap'],
                    'overfitting_severity': self._categorize_overfitting(model_info['overfitting_gap']),
                    
                    # Test Objectives
                    'primary_test_objective': self._get_primary_objective(dataset_name, model_name),
                    'secondary_objectives': self._get_secondary_objectives(dataset_name, model_name),
                    
                    # Expected Outcomes
                    'expected_accuracy_range': self._get_expected_accuracy_range(model_info['test_accuracy']),
                    'expected_precision_range': self._get_expected_precision_range(imbalance_ratio),
                    'expected_recall_range': self._get_expected_recall_range(imbalance_ratio),
                    'expected_f1_range': self._get_expected_f1_range(model_info['test_accuracy']),
                    'expected_auc_range': self._get_expected_auc_range(model_info['test_accuracy']),
                    
                    # Feature Analysis Expectations
                    'expected_top_features': self._get_expected_top_features(dataset_name),
                    'feature_importance_validation': self._get_feature_validation_criteria(dataset_name),
                    'shap_analysis_focus': self._get_shap_focus_areas(dataset_name),
                    
                    # UI/UX Validation Points
                    'ui_load_time_expectation': '< 5 seconds',
                    'chart_rendering_check': 'All charts display correctly',
                    'data_accuracy_check': 'Metrics match backend calculations',
                    'responsiveness_check': 'UI responsive on different screen sizes',
                    
                    # Threshold Analysis
                    'optimal_threshold_expectation': self._get_threshold_expectation(imbalance_ratio),
                    'threshold_sensitivity_check': 'Precision-recall tradeoff visible',
                    
                    # ROC Analysis
                    'roc_curve_expectation': self._get_roc_expectation(model_info['test_accuracy']),
                    'auc_interpretation': self._get_auc_interpretation(model_info['test_accuracy']),
                    
                    # AI Explanation Tests
                    'ai_explanation_trigger': 'Test AI explanation button functionality',
                    'explanation_relevance': 'AI explanations should be contextually relevant',
                    'explanation_accuracy': 'AI should correctly interpret the metrics',
                    
                    # Edge Cases to Test
                    'edge_cases': self._get_edge_cases(dataset_name, model_name),
                    
                    # Success Criteria
                    'pass_criteria': self._get_pass_criteria(model_info['overfitting_gap'], imbalance_ratio),
                    'fail_criteria': self._get_fail_criteria(),
                    
                    # Notes and Observations
                    'testing_notes': f"Focus on {self._get_testing_focus(dataset_name, model_name)}",
                    'expected_challenges': self._get_expected_challenges(dataset_name, model_name),
                    
                    # File Paths for Testing
                    'model_file_path': model_info['model_path'],
                    'scaler_file_path': model_info.get('scaler_path', 'N/A'),
                    'train_dataset_path': f"test_datasets/{dataset_name.lower().replace(' ', '_')}/train_dataset.csv",
                    'test_dataset_path': f"test_datasets/{dataset_name.lower().replace(' ', '_')}/test_dataset.csv",
                    
                    # Test Status Tracking
                    'test_status': 'Not Started',
                    'test_date': '',
                    'tester_name': '',
                    'actual_results': '',
                    'issues_found': '',
                    'test_verdict': ''
                }
                
                test_cases.append(test_case)
        
        return test_cases
    
    def _categorize_dataset(self, dataset_name):
        """Categorize the dataset type"""
        if 'Financial' in dataset_name:
            return 'High-dimensional Balanced'
        elif 'Medical' in dataset_name:
            return 'Imbalanced Binary'
        elif 'Industrial' in dataset_name:
            return 'Non-linear Complex'
        elif 'Noisy' in dataset_name:
            return 'Low Signal-to-Noise'
        elif 'Customer' in dataset_name:
            return 'Mixed Categorical-Numerical'
        return 'General'
    
    def _assess_complexity(self, dataset_name, n_features):
        """Assess the complexity level of the dataset"""
        if 'Noisy' in dataset_name or 'Industrial' in dataset_name:
            return 'High'
        elif n_features > 15:
            return 'Medium-High'
        elif n_features > 10:
            return 'Medium'
        else:
            return 'Low-Medium'
    
    def _categorize_overfitting(self, gap):
        """Categorize the severity of overfitting"""
        if gap < 0.05:
            return 'None/Minimal'
        elif gap < 0.10:
            return 'Mild'
        elif gap < 0.20:
            return 'Moderate'
        else:
            return 'Severe'
    
    def _get_primary_objective(self, dataset_name, model_name):
        """Get the primary testing objective"""
        objectives = {
            'Financial': 'Test feature importance ranking and model interpretability',
            'Medical': 'Test precision-recall balance and threshold optimization',
            'Industrial': 'Test complex feature interaction detection',
            'Noisy': 'Test model robustness with irrelevant features',
            'Customer': 'Test mixed feature type handling'
        }
        
        for key, objective in objectives.items():
            if key in dataset_name:
                if 'overfitted' in model_name:
                    return f"{objective} + Overfitting detection"
                return objective
        
        return 'General classification performance validation'
    
    def _get_secondary_objectives(self, dataset_name, model_name):
        """Get secondary testing objectives"""
        objectives = []
        
        if 'overfitted' in model_name:
            objectives.append('Validate overfitting warning systems')
        
        if 'Medical' in dataset_name:
            objectives.extend(['Test false positive/negative analysis', 'Validate threshold sensitivity'])
        
        if 'Industrial' in dataset_name:
            objectives.extend(['Test SHAP interaction plots', 'Validate complex feature explanations'])
        
        objectives.extend([
            'Test AI explanation generation',
            'Validate chart rendering performance',
            'Test responsive design elements'
        ])
        
        return '; '.join(objectives)
    
    def _get_expected_accuracy_range(self, test_accuracy):
        """Get expected accuracy range based on model performance"""
        lower = max(0.5, test_accuracy - 0.05)
        upper = min(1.0, test_accuracy + 0.05)
        return f"{lower:.2f} - {upper:.2f}"
    
    def _get_expected_precision_range(self, imbalance_ratio):
        """Get expected precision range based on class imbalance"""
        if imbalance_ratio > 3:
            return "0.60 - 0.85"
        elif imbalance_ratio > 2:
            return "0.70 - 0.90"
        else:
            return "0.75 - 0.95"
    
    def _get_expected_recall_range(self, imbalance_ratio):
        """Get expected recall range based on class imbalance"""
        if imbalance_ratio > 3:
            return "0.50 - 0.80"
        elif imbalance_ratio > 2:
            return "0.65 - 0.85"
        else:
            return "0.70 - 0.90"
    
    def _get_expected_f1_range(self, test_accuracy):
        """Get expected F1 range"""
        lower = max(0.5, test_accuracy - 0.08)
        upper = min(1.0, test_accuracy + 0.02)
        return f"{lower:.2f} - {upper:.2f}"
    
    def _get_expected_auc_range(self, test_accuracy):
        """Get expected AUC range"""
        lower = max(0.5, test_accuracy - 0.03)
        upper = min(1.0, test_accuracy + 0.05)
        return f"{lower:.2f} - {upper:.2f}"
    
    def _get_expected_top_features(self, dataset_name):
        """Get expected top features for each dataset"""
        top_features = {
            'Financial': 'income_score, credit_history, debt_ratio, savings_amount',
            'Medical': 'blood_pressure, cholesterol_level, glucose_level, bmi_score',
            'Industrial': 'sensor features and interaction terms',
            'Noisy': 'First 5 features should dominate',
            'Customer': 'numerical features and encoded categories'
        }
        
        for key, features in top_features.items():
            if key in dataset_name:
                return features
        
        return 'Feature importance should follow logical patterns'
    
    def _get_feature_validation_criteria(self, dataset_name):
        """Get feature validation criteria"""
        if 'Financial' in dataset_name:
            return 'Financial features should rank higher than demographic ones'
        elif 'Medical' in dataset_name:
            return 'Clinical measurements should have highest importance'
        elif 'Noisy' in dataset_name:
            return 'Only informative features should have significant importance'
        else:
            return 'Feature importance should be logically distributed'
    
    def _get_shap_focus_areas(self, dataset_name):
        """Get SHAP analysis focus areas"""
        focus_areas = {
            'Financial': 'Individual prediction explanations for edge cases',
            'Medical': 'Feature contributions for high-risk predictions',
            'Industrial': 'Feature interaction plots and dependence analysis',
            'Noisy': 'Verify noise features have minimal SHAP values',
            'Customer': 'Categorical feature impact analysis'
        }
        
        for key, focus in focus_areas.items():
            if key in dataset_name:
                return focus
        
        return 'General SHAP value interpretation'
    
    def _get_threshold_expectation(self, imbalance_ratio):
        """Get threshold analysis expectations"""
        if imbalance_ratio > 3:
            return 'Optimal threshold should be < 0.5 (favoring recall)'
        elif imbalance_ratio > 2:
            return 'Optimal threshold around 0.4-0.6'
        else:
            return 'Optimal threshold around 0.5'
    
    def _get_roc_expectation(self, test_accuracy):
        """Get ROC curve expectations"""
        if test_accuracy > 0.85:
            return 'ROC curve should bow strongly toward top-left'
        elif test_accuracy > 0.75:
            return 'ROC curve should show good separation from diagonal'
        else:
            return 'ROC curve should be moderately above diagonal'
    
    def _get_auc_interpretation(self, test_accuracy):
        """Get AUC interpretation guidelines"""
        if test_accuracy > 0.85:
            return 'Excellent discrimination (AUC > 0.85)'
        elif test_accuracy > 0.75:
            return 'Good discrimination (AUC 0.75-0.85)'
        else:
            return 'Fair discrimination (AUC 0.65-0.75)'
    
    def _get_edge_cases(self, dataset_name, model_name):
        """Get edge cases to test"""
        edge_cases = []
        
        if 'overfitted' in model_name:
            edge_cases.append('Check if overfitting warning appears')
        
        if 'Medical' in dataset_name:
            edge_cases.extend(['Test with extreme threshold values', 'Check false positive implications'])
        
        edge_cases.extend([
            'Test with different browser sizes',
            'Check chart zoom and interaction features',
            'Verify AI explanation with complex cases'
        ])
        
        return '; '.join(edge_cases)
    
    def _get_pass_criteria(self, overfitting_gap, imbalance_ratio):
        """Define pass criteria"""
        criteria = [
            'All metrics display correctly',
            'Charts render without errors',
            'AI explanations generate successfully'
        ]
        
        if overfitting_gap > 0.1:
            criteria.append('Overfitting indicators should be visible')
        
        if imbalance_ratio > 2:
            criteria.append('Precision-recall tradeoff should be evident')
        
        return '; '.join(criteria)
    
    def _get_fail_criteria(self):
        """Define fail criteria"""
        return 'Any crashes, incorrect calculations, missing charts, or non-responsive UI elements'
    
    def _get_testing_focus(self, dataset_name, model_name):
        """Get specific testing focus"""
        if 'overfitted' in model_name:
            return 'overfitting detection and warnings'
        elif 'Medical' in dataset_name:
            return 'precision-recall optimization'
        elif 'Industrial' in dataset_name:
            return 'feature interaction analysis'
        else:
            return 'general performance metrics and interpretability'
    
    def _get_expected_challenges(self, dataset_name, model_name):
        """Get expected testing challenges"""
        challenges = []
        
        if 'Noisy' in dataset_name:
            challenges.append('Many features may show minimal importance')
        
        if 'overfitted' in model_name:
            challenges.append('Large performance gap between train/test')
        
        if 'Medical' in dataset_name:
            challenges.append('Class imbalance may affect threshold selection')
        
        if not challenges:
            challenges.append('Standard classification analysis')
        
        return '; '.join(challenges)
    
    def generate_functional_test_cases(self):
        """Generate functional UI test cases"""
        functional_tests = [
            {
                'test_id': 'UI_001',
                'test_category': 'Upload Functionality',
                'test_description': 'Upload model and dataset files',
                'test_steps': '1. Navigate to upload page\n2. Select model file\n3. Select train dataset\n4. Select test dataset\n5. Enter target column\n6. Click upload',
                'expected_result': 'Files upload successfully, redirect to classification stats',
                'priority': 'High',
                'test_data': 'Any valid model and dataset combination'
            },
            {
                'test_id': 'UI_002',
                'test_category': 'Classification Stats Display',
                'test_description': 'Verify all classification metrics display',
                'test_steps': '1. Upload model and data\n2. Navigate to classification stats\n3. Check all metric cards\n4. Verify confusion matrix\n5. Check ROC curve',
                'expected_result': 'All metrics display with correct values',
                'priority': 'High',
                'test_data': 'Financial approval dataset'
            },
            {
                'test_id': 'UI_003',
                'test_category': 'Threshold Analysis',
                'test_description': 'Test threshold analysis functionality',
                'test_steps': '1. Navigate to classification stats\n2. Locate threshold analysis chart\n3. Verify precision-recall curves\n4. Check optimal threshold marking',
                'expected_result': 'Threshold analysis chart displays with interactive elements',
                'priority': 'Medium',
                'test_data': 'Medical disease risk dataset'
            },
            {
                'test_id': 'UI_004',
                'test_category': 'AI Explanation',
                'test_description': 'Test AI explanation generation',
                'test_steps': '1. Click "Explain with AI" button\n2. Wait for explanation\n3. Verify explanation content\n4. Check explanation relevance',
                'expected_result': 'AI generates relevant explanation of the classification results',
                'priority': 'Medium',
                'test_data': 'Any dataset'
            },
            {
                'test_id': 'UI_005',
                'test_category': 'Responsive Design',
                'test_description': 'Test UI responsiveness',
                'test_steps': '1. Open app in different screen sizes\n2. Check mobile view\n3. Verify chart scaling\n4. Test navigation on small screens',
                'expected_result': 'UI adapts correctly to different screen sizes',
                'priority': 'Low',
                'test_data': 'Any dataset'
            },
            {
                'test_id': 'UI_006',
                'test_category': 'Error Handling',
                'test_description': 'Test error handling for invalid inputs',
                'test_steps': '1. Upload invalid model file\n2. Upload mismatched datasets\n3. Enter wrong target column\n4. Check error messages',
                'expected_result': 'Clear error messages displayed for all invalid inputs',
                'priority': 'High',
                'test_data': 'Invalid/corrupted files'
            }
        ]
        
        return functional_tests
    
    def generate_performance_test_cases(self):
        """Generate performance test cases"""
        performance_tests = [
            {
                'test_id': 'PERF_001',
                'test_category': 'Load Time',
                'test_description': 'Measure classification stats load time',
                'metric': 'Page load time',
                'target': '< 5 seconds',
                'test_data': 'Large dataset (2000+ samples)',
                'measurement_method': 'Browser developer tools'
            },
            {
                'test_id': 'PERF_002',
                'test_category': 'Chart Rendering',
                'test_description': 'Measure chart rendering performance',
                'metric': 'Chart render time',
                'target': '< 2 seconds',
                'test_data': 'Complex ROC and threshold charts',
                'measurement_method': 'Performance timeline'
            },
            {
                'test_id': 'PERF_003',
                'test_category': 'AI Explanation',
                'test_description': 'Measure AI explanation generation time',
                'metric': 'AI response time',
                'target': '< 10 seconds',
                'test_data': 'Complex dataset with many features',
                'measurement_method': 'Network tab timing'
            }
        ]
        
        return performance_tests
    
    def generate_comprehensive_documentation(self):
        """Generate comprehensive test documentation"""
        print("Generating comprehensive test documentation...")
        
        # Generate main test cases
        test_cases = self.generate_test_cases()
        functional_tests = self.generate_functional_test_cases()
        performance_tests = self.generate_performance_test_cases()
        
        # Create DataFrames
        test_cases_df = pd.DataFrame(test_cases)
        functional_df = pd.DataFrame(functional_tests)
        performance_df = pd.DataFrame(performance_tests)
        
        # Generate summary statistics
        summary_stats = self._generate_summary_stats(test_cases)
        summary_df = pd.DataFrame([summary_stats])
        
        # Save to Excel with multiple sheets
        excel_path = os.path.join(self.output_dir, f'classification_stats_test_documentation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Main test cases
            test_cases_df.to_excel(writer, sheet_name='Main_Test_Cases', index=False)
            
            # Functional tests
            functional_df.to_excel(writer, sheet_name='Functional_Tests', index=False)
            
            # Performance tests
            performance_df.to_excel(writer, sheet_name='Performance_Tests', index=False)
            
            # Summary
            summary_df.to_excel(writer, sheet_name='Test_Summary', index=False)
        
        # Also save individual CSV files
        test_cases_df.to_csv(os.path.join(self.output_dir, 'main_test_cases.csv'), index=False)
        functional_df.to_csv(os.path.join(self.output_dir, 'functional_test_cases.csv'), index=False)
        performance_df.to_csv(os.path.join(self.output_dir, 'performance_test_cases.csv'), index=False)
        
        # Generate test execution template
        self._generate_test_execution_template()
        
        print(f"✓ Test documentation generated successfully!")
        print(f"📁 Excel file: {excel_path}")
        print(f"📁 CSV files: {self.output_dir}/")
        
        return {
            'excel_path': excel_path,
            'total_test_cases': len(test_cases),
            'functional_tests': len(functional_tests),
            'performance_tests': len(performance_tests),
            'datasets_covered': len(set(tc['dataset_name'] for tc in test_cases)),
            'models_covered': len(set(tc['model_name'] for tc in test_cases))
        }
    
    def _generate_summary_stats(self, test_cases):
        """Generate summary statistics"""
        return {
            'total_test_cases': len(test_cases),
            'datasets_covered': len(set(tc['dataset_name'] for tc in test_cases)),
            'models_covered': len(set(tc['model_name'] for tc in test_cases)),
            'high_complexity_cases': len([tc for tc in test_cases if tc['complexity_level'] == 'High']),
            'overfitting_cases': len([tc for tc in test_cases if 'overfitted' in tc['model_name']]),
            'imbalanced_cases': len([tc for tc in test_cases if tc['imbalance_ratio'] > 2]),
            'expected_duration_hours': len(test_cases) * 0.5,  # 30 min per test case
            'recommended_testers': max(2, len(test_cases) // 10)
        }
    
    def _generate_test_execution_template(self):
        """Generate a test execution tracking template"""
        template = {
            'Test Execution Tracking': [
                {
                    'date': '',
                    'tester_name': '',
                    'test_environment': 'Local/Staging/Production',
                    'browser': 'Chrome/Firefox/Safari/Edge',
                    'screen_resolution': '1920x1080/1366x768/Mobile',
                    'test_session_notes': '',
                    'overall_pass_rate': '',
                    'critical_issues_found': '',
                    'recommendations': ''
                }
            ]
        }
        
        template_df = pd.DataFrame(template['Test Execution Tracking'])
        template_df.to_csv(os.path.join(self.output_dir, 'test_execution_template.csv'), index=False)

def main():
    """Generate comprehensive test documentation"""
    print("=" * 60)
    print("GENERATING TEST DOCUMENTATION FOR CLASSIFICATION STATS")
    print("=" * 60)
    
    generator = TestDocumentationGenerator()
    results = generator.generate_comprehensive_documentation()
    
    print("\n" + "=" * 60)
    print("DOCUMENTATION GENERATION COMPLETE!")
    print("=" * 60)
    print(f"📊 Total test cases: {results['total_test_cases']}")
    print(f"📁 Datasets covered: {results['datasets_covered']}")
    print(f"🤖 Models covered: {results['models_covered']}")
    print(f"⚡ Functional tests: {results['functional_tests']}")
    print(f"🚀 Performance tests: {results['performance_tests']}")
    print(f"📋 Excel file: {results['excel_path']}")
    
    print("\n📝 TEST EXECUTION GUIDELINES:")
    print("1. Review the 'Main_Test_Cases' sheet for detailed test scenarios")
    print("2. Execute 'Functional_Tests' for UI/UX validation")
    print("3. Run 'Performance_Tests' to ensure speed requirements")
    print("4. Use 'test_execution_template.csv' to track progress")
    print("5. Document all findings in the respective test case rows")
    
    return results

if __name__ == "__main__":
    main()
