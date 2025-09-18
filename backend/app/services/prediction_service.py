import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from .base_model_service import BaseModelService


class PredictionService:
    """
    Service for individual prediction analysis and what-if scenarios.
    """
    
    def __init__(self, base_service: BaseModelService):
        self.base = base_service

    def individual_prediction(self, instance_idx: int) -> Dict[str, Any]:
        """Get detailed prediction analysis for a single instance."""
        self.base._is_ready()
        
        if not (0 <= instance_idx < len(self.base.X_df)):
            raise ValueError(f"Instance index {instance_idx} is out of range. Dataset has {len(self.base.X_df)} instances.")
        
        instance_data = self.base.X_df.iloc[instance_idx]
        instance_df = instance_data.to_frame().T
        proba = float(self.base.model.predict_proba(instance_df)[0, 1])
        confidence = max(proba, 1.0 - proba)
        shap_vals_for_instance = self.base._get_instance_shap_vector(instance_idx)
        
        # Handle case where explainer might not be available
        base_value = 0.0
        if self.base.explainer and hasattr(self.base.explainer, 'expected_value'):
            if isinstance(self.base.explainer.expected_value, (list, np.ndarray)):
                base_value = float(self.base.explainer.expected_value[1] if len(self.base.explainer.expected_value) > 1 else self.base.explainer.expected_value[0])
            else:
                base_value = float(self.base.explainer.expected_value)

        contributions = [
            {
                "name": name,
                "value": self.base._safe_float(instance_data[name]),
                "shap": float(shap_vals_for_instance[i])
            }
            for i, name in enumerate(self.base.feature_names)
        ]
        contributions.sort(key=lambda x: abs(x["shap"]), reverse=True)

        return {
            "prediction_percentage": proba * 100.0,
            "actual_outcome": int(self.base.y_s.iloc[instance_idx]),
            "confidence_score": confidence,
            "base_value": base_value,
            "shap_values": [float(v) for v in shap_vals_for_instance],
            "feature_contributions": contributions
        }

    def explain_instance(self, instance_idx: int) -> Dict[str, Any]:
        """Explain a single instance prediction with detailed SHAP analysis."""
        self.base._is_ready()
        
        if not (0 <= instance_idx < len(self.base.X_df)):
            raise ValueError(f"Instance index {instance_idx} is out of range. Dataset has {len(self.base.X_df)} instances.")
            
        instance_data = self.base.X_df.iloc[instance_idx]
        shap_vals_for_instance = self.base._get_instance_shap_vector(instance_idx)
        
        # Handle case where explainer might not be available
        base_value = 0.0
        if self.base.explainer and hasattr(self.base.explainer, 'expected_value'):
            if isinstance(self.base.explainer.expected_value, (list, np.ndarray)):
                base_value = float(self.base.explainer.expected_value[1] if len(self.base.explainer.expected_value) > 1 else self.base.explainer.expected_value[0])
            else:
                base_value = float(self.base.explainer.expected_value)

        # Create single-row DataFrame for prediction to maintain feature names
        instance_df = pd.DataFrame([instance_data], columns=self.base.feature_names)
        # Use .values to avoid feature name warnings
        # Get model prediction
        prediction_prob = float(self.base.safe_predict_proba(instance_df)[0, 1])

        # Prepare both mapping and ordered arrays for convenience on the frontend
        shap_mapping = dict(zip(self.base.feature_names, shap_vals_for_instance))
        ordered = sorted(shap_mapping.items(), key=lambda kv: abs(kv[1]), reverse=True)
        ordered_features = [name for name, _ in ordered]
        ordered_values = [float(val) for _, val in ordered]
        ordered_feature_values = [instance_data[name] for name in ordered_features]

        return {
            "instance_id": instance_idx,
            "features": instance_data.to_dict(),
            "prediction": prediction_prob,
            "actual_value": int(self.base.y_s.iloc[instance_idx]),
            "base_value": float(base_value),
            "shap_values_map": shap_mapping,
            "ordered_contributions": {
                "feature_names": ordered_features,
                "feature_values": [self.base._safe_float(v) for v in ordered_feature_values],
                "shap_values": ordered_values
            }
        }

    def perform_what_if(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Perform what-if analysis by modifying feature values."""
        self.base._is_ready()
        
        try:
            # Start with the first instance as base
            base_instance = self.base.X_df.iloc[0].copy()
            
            # Update with provided feature values
            for feature_name, value in features.items():
                if feature_name in base_instance.index:
                    base_instance[feature_name] = value
                else:
                    raise ValueError(f"Feature '{feature_name}' not found in dataset.")
            
            # Make prediction
            instance_df = pd.DataFrame([base_instance], columns=self.base.feature_names)
            prediction_proba = self.base.safe_predict_proba(instance_df)[0]
            prediction_prob = float(prediction_proba[1])  # Positive class probability
            prediction_class = int(np.argmax(prediction_proba))
            
            # Get SHAP explanation if available
            shap_explanation = {}
            if self.base.explainer:
                try:
                    shap_values = self.base.explainer.shap_values(instance_df)
                    if isinstance(shap_values, list):
                        shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                    else:
                        if len(shap_values.shape) == 3:
                            shap_vals = shap_values[0, :, 1] if shap_values.shape[2] > 1 else shap_values[0, :, 0]
                        else:
                            shap_vals = shap_values[0]
                    
                    shap_explanation = dict(zip(self.base.feature_names, [float(v) for v in shap_vals]))
                except Exception as e:
                    print(f"SHAP explanation failed: {e}")
            
            return {
                "modified_features": features,
                "prediction_probability": prediction_prob,
                "predicted_class": prediction_class,
                "feature_values": base_instance.to_dict(),
                "shap_explanations": shap_explanation,
                "feature_ranges": self._get_feature_ranges()
            }
            
        except Exception as e:
            raise ValueError(f"What-if analysis failed: {str(e)}")

    def _get_feature_ranges(self) -> Dict[str, Any]:
        """Get feature ranges and metadata for what-if analysis."""
        self.base._is_ready()
        
        feature_ranges = {}
        for feature_name in self.base.feature_names:
            col = self.base.X_df[feature_name]
            is_numeric = pd.api.types.is_numeric_dtype(col.dtype)
            
            if is_numeric:
                feature_ranges[feature_name] = {
                    "type": "numeric",
                    "min": float(col.min()),
                    "max": float(col.max()),
                    "mean": float(col.mean()),
                    "std": float(col.std()),
                    "median": float(col.median()),
                    "step": self._calculate_step(col)
                }
            else:
                # For categorical features, provide the most common categories
                value_counts = col.value_counts()
                feature_ranges[feature_name] = {
                    "type": "categorical",
                    "categories": value_counts.index.tolist(),
                    "frequencies": value_counts.values.tolist(),
                    "most_common": value_counts.index[0] if len(value_counts) > 0 else None
                }
        
        return feature_ranges
    
    def _calculate_step(self, column: pd.Series) -> float:
        """Calculate appropriate step size for numeric column."""
        col_range = column.max() - column.min()
        
        # For very small ranges (< 1), use smaller steps
        if col_range < 1:
            return 0.01
        # For medium ranges (1-100), use 0.1 or 1
        elif col_range < 100:
            return 0.1 if col_range < 10 else 1
        # For large ranges, use larger steps
        elif col_range < 1000:
            return 10
        else:
            return 100
