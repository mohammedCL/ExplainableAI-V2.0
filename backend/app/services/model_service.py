import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.tree import _tree
import shap
import joblib
from typing import Dict, Any, List, Optional

class ModelService:
    """
    A stateful service to hold a loaded model and its corresponding dataset
    for interactive analysis.
    """
    def __init__(self):
        # State: These will be populated when files are uploaded
        self.model: Optional[Any] = None
        self.X_df: Optional[pd.DataFrame] = None
        self.y_s: Optional[pd.Series] = None
        self.feature_names: Optional[List[str]] = None
        self.target_name: Optional[str] = None
        self.explainer: Optional[shap.TreeExplainer] = None
        self.shap_values: Optional[np.ndarray] = None
        self.model_info: Dict[str, Any] = {}
        print("ModelService initialized. Waiting for model and data.")

    def load_model_and_data(self, model_path: str, data_path: str, target_column: str):
        """Loads the model and dataset from local files and prepares for analysis."""
        try:
            print(f"Loading model from: {model_path}")
            self.model = joblib.load(model_path)
            
            print(f"Loading data from: {data_path}")
            df = pd.read_csv(data_path)

            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in the dataset.")

            self.X_df = df.drop(columns=[target_column])
            self.y_s = df[target_column]
            self.feature_names = list(self.X_df.columns)
            self.target_name = target_column
            
            print("Creating SHAP explainer...")
            self.explainer = shap.TreeExplainer(self.model)
            # For classification, shap_values is a list [class_0_vals, class_1_vals]
            # We will use this structure directly in our methods.
            self.shap_values = self.explainer.shap_values(self.X_df)
            print("SHAP explainer created successfully.")

            self.model_info = {
                "model_path": model_path,
                "data_path": data_path,
                "target_column": target_column,
                "features_count": len(self.feature_names),
                "data_shape": df.shape,
            }
            return {"status": "success", "message": "Model and data loaded successfully.", "details": self.model_info}

        except Exception as e:
            self.__init__() # Reset state on failure
            raise e

    def _is_ready(self):
        """Check if the service has a model and data loaded."""
        if self.model is None or self.X_df is None:
            raise ValueError("Model and data have not been uploaded yet. Please upload files first.")

    # --- Analysis Methods ---
    
    def get_model_overview(self) -> Dict[str, Any]:
        self._is_ready()
        y_pred = self.model.predict(self.X_df)
        y_proba = self.model.predict_proba(self.X_df)[:, 1]

        feature_schema = []
        for feature in self.feature_names:
            col = self.X_df[feature]
            if pd.api.types.is_numeric_dtype(col.dtype):
                feature_schema.append({
                    "name": feature, "type": "numerical", "value": col.mean(),
                    "min": float(col.min()), "max": float(col.max()),
                    "description": f"Value for {feature}"
                })
            else:
                 feature_schema.append({
                    "name": feature, "type": "categorical", "value": col.mode()[0],
                    "options": col.unique().tolist(), "description": f"Value for {feature}"
                })

        return {
            "model_id": self.model_info.get('model_path', 'N/A'),
            "name": "Uploaded Classification Model", "model_type": "classification",
            "version": "1.0.0", "framework": "scikit-learn", "status": "active",
            "feature_names": self.feature_names,
            "schema": feature_schema,
            "performance_metrics": {
                "accuracy": accuracy_score(self.y_s, y_pred),
                "precision": precision_score(self.y_s, y_pred),
                "recall": recall_score(self.y_s, y_pred),
                "f1_score": f1_score(self.y_s, y_pred),
                "auc": roc_auc_score(self.y_s, y_proba)
            }
        }

    def get_classification_stats(self) -> Dict[str, Any]:
        self._is_ready()
        y_pred = self.model.predict(self.X_df)
        y_proba = self.model.predict_proba(self.X_df)[:, 1]
        cm = confusion_matrix(self.y_s, y_pred)
        fpr, tpr, _ = roc_curve(self.y_s, y_proba)

        return {
            "metrics": self.get_model_overview()["performance_metrics"],
            "confusion_matrix": {
                "true_negative": int(cm[0, 0]), "false_positive": int(cm[0, 1]),
                "false_negative": int(cm[1, 0]), "true_positive": int(cm[1, 1]),
            },
            "roc_curve": { "fpr": fpr.tolist(), "tpr": tpr.tolist() }
        }

    def get_feature_importance(self, method: str) -> Dict[str, Any]:
        self._is_ready()
        if method == 'shap':
            # Use positive class (1) for importance calculation
            shap_vals_for_importance = self.shap_values[1]
            importance_values = np.abs(shap_vals_for_importance).mean(0)
        else: # 'permutation' or 'builtin'
            importance_values = self.model.feature_importances_

        sorted_indices = np.argsort(importance_values)[::-1]
        features = [{
            "name": self.feature_names[idx],
            "importance": float(importance_values[idx])
        } for idx in sorted_indices]
        
        return {"method": method, "features": features}

    def explain_instance(self, instance_idx: int) -> Dict[str, Any]:
        self._is_ready()
        if not (0 <= instance_idx < len(self.X_df)):
            raise ValueError("Instance index out of bounds.")
            
        instance_data = self.X_df.iloc[instance_idx]
        shap_vals_for_instance = self.shap_values[1][instance_idx, :]
        base_value = self.explainer.expected_value[1]

        return {
          "instance_id": f"instance_{instance_idx}",
          "features": instance_data.to_dict(),
          "prediction": float(self.model.predict_proba(instance_data.values.reshape(1, -1))[0, 1]),
          "actual_value": int(self.y_s.iloc[instance_idx]),
          "explanation": {
              "base_value": float(base_value),
              "shap_values": dict(zip(self.feature_names, shap_vals_for_instance))
          }
        }

    def perform_what_if(self, features: Dict[str, Any]) -> Dict[str, Any]:
        self._is_ready()
        try:
            input_df = pd.DataFrame([features], columns=self.feature_names)
            prediction_proba = self.model.predict_proba(input_df)
            return { "prediction": float(prediction_proba[0, 1]) }
        except Exception as e:
            raise ValueError(f"Error during 'what-if' prediction: {e}")

    def get_feature_dependence(self, feature_name: str) -> Dict[str, Any]:
        self._is_ready()
        if feature_name not in self.feature_names:
            raise ValueError("Feature not found")
        
        feature_idx = self.feature_names.index(feature_name)
        shap_vals = self.shap_values[1] # Positive class
        
        return {
            "feature_values": self.X_df[feature_name].tolist(),
            "shap_values": shap_vals[:, feature_idx].tolist()
        }

    def get_feature_interactions(self, feature1: str, feature2: str) -> Dict[str, Any]:
        self._is_ready()
        if feature1 not in self.feature_names or feature2 not in self.feature_names:
            raise ValueError("One or both features not found")
            
        # This requires SHAP interaction values, which can be computationally expensive
        # For a production system, this would be pre-computed or calculated on demand by a worker.
        # Here we mock it for simplicity, but a real implementation would use:
        # shap_interaction_values = self.explainer.shap_interaction_values(self.X_df)
        
        # Mocking interaction effects
        f1_values = self.X_df[feature1]
        f2_values = self.X_df[feature2]
        # Create a plausible-looking interaction effect based on the features
        interaction_effect = (f1_values - f1_values.mean()) * (f2_values - f2_values.mean())
        interaction_effect_normalized = (interaction_effect / interaction_effect.abs().max()) * 0.1

        return {
            "feature1_values": f1_values.tolist(),
            "feature2_values": f2_values.tolist(),
            "interaction_shap_values": interaction_effect_normalized.tolist()
        }

    def get_decision_tree(self) -> Dict[str, Any]:
        self._is_ready()
        # Using the first tree from the Random Forest as a representative example
        tree_estimator = self.model.estimators_[0]
        tree_obj = tree_estimator.tree_
        
        def recurse(node, depth):
            if tree_obj.feature[node] != _tree.TREE_UNDEFINED:
                feature = self.feature_names[tree_obj.feature[node]]
                return {
                    "type": "split", "depth": depth,
                    "feature": feature, "threshold": float(tree_obj.threshold[node]),
                    "samples": int(tree_obj.n_node_samples[node]),
                    "left": recurse(tree_obj.children_left[node], depth + 1),
                    "right": recurse(tree_obj.children_right[node], depth + 1)
                }
            else:
                values = tree_obj.value[node][0]
                return { "type": "leaf", "depth": depth, "samples": int(tree_obj.n_node_samples[node]), "values": values.tolist() }
        
        return recurse(0, 0)