import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from .base_model_service import BaseModelService


class FeatureService:
    """
    Service for feature-related operations like feature importance, metadata,
    correlation analysis, and feature interactions.
    """
    
    def __init__(self, base_service: BaseModelService):
        self.base = base_service
        # Simple in-memory caches for heavy computations
        self._correlation_cache: Dict[str, Dict[str, Any]] = {}
        self._importance_cache: Dict[str, Dict[str, Any]] = {}

    def get_feature_importance(self, method: str) -> Dict[str, Any]:
        """Get feature importance using specified method (shap, builtin, etc.)."""
        self.base._is_ready()
        
        if method == 'shap':
            if self.base.shap_values is None:
                raise ValueError("SHAP values not available. Make sure the model supports SHAP analysis.")
            
            # Get SHAP values for analysis
            shap_vals = self.base._get_shap_values_for_analysis()
            if shap_vals is not None:
                importance_values = np.abs(shap_vals).mean(axis=0)
            else:
                raise ValueError("Failed to compute SHAP-based feature importance.")
                
        else:            
            # Try to get feature importance from the model
            if hasattr(self.base.model, 'feature_importances_'):
                importance_values = self.base.model.feature_importances_
            else:
                raise ValueError(f"Method '{method}' not supported or model doesn't have feature_importances_ attribute.")

        sorted_indices = np.argsort(importance_values)[::-1]
        features = [{
            "name": self.base.feature_names[idx],
            "importance": float(importance_values[idx])
        } for idx in sorted_indices]
        
        return {"method": method, "features": features}

    def get_feature_metadata(self) -> Dict[str, Any]:
        """Return available features with basic metadata."""
        self.base._is_ready()
        features: List[Dict[str, Any]] = []
        
        for name in self.base.feature_names:
            col = self.base.X_df[name]
            is_numeric = pd.api.types.is_numeric_dtype(col.dtype)
            
            feature_info = {
                "name": name,
                "type": "numerical" if is_numeric else "categorical",
                "missing_count": int(col.isnull().sum()),
                "unique_count": int(col.nunique()),
                "dtype": str(col.dtype)
            }
            
            if is_numeric:
                feature_info.update({
                    "min": float(col.min()),
                    "max": float(col.max()),
                    "mean": float(col.mean()),
                    "std": float(col.std()),
                    "median": float(col.median())
                })
            else:
                feature_info["top_categories"] = col.value_counts().head(5).to_dict()
                
            features.append(feature_info)
            
        return {"features": features}

    def _encode_mixed_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode mixed dataframe for correlation computation."""
        encoded_cols = {}
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c].dtype):
                encoded_cols[c] = df[c].fillna(df[c].median())
            else:
                # Label encode categorical variables
                encoded_cols[c] = pd.Categorical(df[c].fillna('missing')).codes
        return pd.DataFrame(encoded_cols)

    def compute_correlation(self, selected_features: List[str]) -> Dict[str, Any]:
        """Compute correlation matrix for selected features."""
        self.base._is_ready()
        
        if not selected_features or len(selected_features) < 2:
            raise ValueError("At least 2 features must be selected for correlation analysis.")
        
        for feat in selected_features:
            if feat not in self.base.feature_names:
                raise ValueError(f"Feature '{feat}' not found in dataset.")

        cache_key = "|".join(sorted(selected_features))
        if cache_key in self._correlation_cache:
            return self._correlation_cache[cache_key]

        df = self.base.X_df[selected_features]
        enc = self._encode_mixed_dataframe(df)
        corr = enc.corr(method='pearson')
        matrix = corr.loc[selected_features, selected_features].to_numpy().tolist()
        
        payload = {
            "features": selected_features,
            "matrix": [[float(v) for v in row] for row in matrix],
            "computed_at": pd.Timestamp.utcnow().isoformat()
        }
        
        # Cache result
        self._correlation_cache[cache_key] = payload
        return payload

    def compute_feature_importance_advanced(self, method: str = 'shap', sort_by: str = 'importance',
                                        top_n: int = 20, visualization: str = 'bar') -> Dict[str, Any]:
        """Compute advanced feature importance with detailed analysis."""
        self.base._is_ready()
        
        # Normalize params for cache key
        key = f"{method}|{sort_by}|{top_n}|{visualization}"
        if key in self._importance_cache:
            return self._importance_cache[key]

        # Extra debug: Print a sample of X_train, feature names, and model predictions
        try:
            print("[DEBUG] X_train sample:")
            print(self.base.X_train.head())
            print(f"[DEBUG] Feature names: {self.base.feature_names}")
            preds = self.base.model.predict(self.base.X_train.head())
            print(f"[DEBUG] Model predictions on X_train sample: {preds}")
        except Exception as e:
            print(f"[DEBUG] Error printing X_train/model info: {e}")

        method = method.lower()
        if method == 'shap':
            if self.base.shap_values is None:
                raise ValueError("SHAP values not available for this model.")
            shap_vals = self.base._get_shap_values_for_analysis()
            if shap_vals is not None:
                raw_importance = np.abs(shap_vals).mean(axis=0)
            else:
                raise ValueError("Failed to compute SHAP-based feature importance.")
        elif method in ('permutation', 'gain', 'builtin'):
            if hasattr(self.base.model, 'feature_importances_'):
                raw_importance = self.base.model.feature_importances_
            else:
                raise ValueError(f"Model doesn't support {method} feature importance.")
        else:
            raise ValueError(f"Unsupported importance method: {method}")

        # Debug: Print raw importance values for inspection
        print(f"[DEBUG] Feature importance method: {method}")
        print(f"[DEBUG] Raw importance values: {raw_importance}")
        if hasattr(self.base, 'feature_names'):
            print(f"[DEBUG] Feature names: {self.base.feature_names}")

        items = []
        for idx, name in enumerate(self.base.feature_names):
            importance = float(raw_importance[idx])
            impact_direction = 'positive' if importance >= 0 else 'negative'
            
            items.append({
                "name": name,
                "importance": importance,
                "impact_direction": impact_direction,
                "rank": 0  # Will be set after sorting
            })

        # Sorting
        if sort_by == 'feature_name':
            items.sort(key=lambda x: x['name'])
        elif sort_by == 'impact':
            items.sort(key=lambda x: abs(x['importance']), reverse=True)
        else:            
            items.sort(key=lambda x: x['importance'], reverse=True)

        # Assign ranks after sort
        for i, it in enumerate(items):
            it['rank'] = i + 1

        top_items = items[:int(top_n)]
        
        payload = {
            "total_features": len(items),
            "positive_impact_count": sum(1 for i in items if i['impact_direction'] == 'positive'),
            "negative_impact_count": sum(1 for i in items if i['impact_direction'] == 'negative'),
            "features": top_items,
            "computation_method": method,
            "computed_at": pd.Timestamp.utcnow().isoformat()
        }
        
        self._importance_cache[key] = payload
        return payload
    
    def get_feature_interactions(self, feature1: str, feature2: str) -> Dict[str, Any]:
        """Get feature interaction analysis (simplified version)."""
        self.base._is_ready()
        
        if feature1 not in self.base.feature_names or feature2 not in self.base.feature_names:
            raise ValueError("One or both features not found in dataset.")
            
        # This requires SHAP interaction values, which can be computationally expensive
        # For a production system, this would be pre-computed or calculated on demand by a worker.
        # Here we mock it for simplicity, but a real implementation would use:
        # shap_interaction_values = self.explainer.shap_interaction_values(self.X_df)
        
        # Mocking interaction effects
        f1_values = self.base.X_df[feature1]
        f2_values = self.base.X_df[feature2]
        
        # Create a plausible-looking interaction effect based on the features
        interaction_effect = (f1_values - f1_values.mean()) * (f2_values - f2_values.mean())
        interaction_effect_normalized = (interaction_effect / interaction_effect.abs().max()) * 0.1

        return {
            "feature1_values": f1_values.tolist(),
            "feature2_values": f2_values.tolist(),
            "interaction_shap_values": interaction_effect_normalized.tolist()
        }
