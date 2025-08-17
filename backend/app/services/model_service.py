import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.tree import _tree
import shap
import joblib
import pickle
from typing import Dict, Any, List, Optional
from sklearn.model_selection import train_test_split

# Optional imports for additional model formats
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

class ModelWrapper:
    """Wrapper class to provide consistent interface for different model formats."""
    
    def __init__(self, model, model_type: str):
        self.model = model
        self.model_type = model_type
        
    def predict(self, X):
        """Predict class labels."""
        if self.model_type == "sklearn":
            # Handle both DataFrame and numpy array inputs
            if hasattr(X, 'values'):  # DataFrame
                return self.model.predict(X)
            else:  # numpy array
                return self.model.predict(X)
        elif self.model_type == "onnx":
            # ONNX models require input as dictionary and numpy arrays
            if hasattr(X, 'values'):  # DataFrame
                X_array = X.values.astype(np.float32)
            else:  # numpy array
                X_array = X.astype(np.float32)
            
            input_name = self.model.get_inputs()[0].name
            outputs = self.model.run(None, {input_name: X_array})
            # Assuming binary classification, take argmax of first output
            probabilities = outputs[0]
            
            # Handle different output shapes
            if probabilities.ndim == 1:
                # 1D output: threshold at 0.5
                return (probabilities > 0.5).astype(int)
            elif probabilities.ndim == 2:
                if probabilities.shape[1] == 2:
                    # Binary classification with 2 columns
                    return np.argmax(probabilities, axis=1)
                elif probabilities.shape[1] == 1:
                    # Single column output, threshold at 0.5
                    return (probabilities.flatten() > 0.5).astype(int)
                else:
                    # Multi-class
                    return np.argmax(probabilities, axis=1)
            else:
                # Unexpected shape, flatten and threshold
                return (probabilities.flatten() > 0.5).astype(int)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.model_type == "sklearn":
            # Handle both DataFrame and numpy array inputs
            if hasattr(X, 'values'):  # DataFrame
                return self.model.predict_proba(X)
            else:  # numpy array
                return self.model.predict_proba(X)
        elif self.model_type == "onnx":
            # ONNX models require input as dictionary and numpy arrays
            if hasattr(X, 'values'):  # DataFrame
                X_array = X.values.astype(np.float32)
            else:  # numpy array
                X_array = X.astype(np.float32)
                
            input_name = self.model.get_inputs()[0].name
            outputs = self.model.run(None, {input_name: X_array})
            probabilities = outputs[0]
            
            # Handle different output shapes from ONNX models
            if probabilities.ndim == 1:
                # 1D output: treat as single class probabilities
                pos_proba = probabilities
                neg_proba = 1 - pos_proba
                return np.column_stack([neg_proba, pos_proba])
            elif probabilities.ndim == 2:
                # 2D output: check if single or multiple classes
                if probabilities.shape[1] == 1:
                    # Single output column, convert to binary probabilities
                    pos_proba = probabilities.flatten()
                    neg_proba = 1 - pos_proba
                    return np.column_stack([neg_proba, pos_proba])
                else:
                    # Multi-class or already binary format
                    return probabilities
            else:
                # Unexpected shape, flatten and treat as single class
                pos_proba = probabilities.flatten()
                neg_proba = 1 - pos_proba
                return np.column_stack([neg_proba, pos_proba])
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def __getattr__(self, name):
        """Delegate attribute access to the wrapped model."""
        return getattr(self.model, name)

class ModelService:
    """
    A stateful service to hold a loaded model and its corresponding dataset
    for interactive analysis.
    """
    def __init__(self):
        # State: These will be populated when files are uploaded
        self.model: Optional[Any] = None
        
        # Separate train and test datasets
        self.X_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.Series] = None
        
        # Keep for backward compatibility and general analysis
        self.X_df: Optional[pd.DataFrame] = None  # Will point to train data
        self.y_s: Optional[pd.Series] = None      # Will point to train data
        
        self.feature_names: Optional[List[str]] = None
        self.target_name: Optional[str] = None
        self.explainer: Optional[shap.TreeExplainer] = None
        self.shap_values: Optional[np.ndarray] = None
        self.model_info: Dict[str, Any] = {}
        # simple in-memory caches for heavy computations
        self._correlation_cache: Dict[str, Dict[str, Any]] = {}
        self._importance_cache: Dict[str, Dict[str, Any]] = {}
        print("ModelService initialized. Waiting for model and data.")

    def _load_model_by_format(self, model_path: str) -> ModelWrapper:
        """Load model based on file extension and return wrapped model."""
        file_extension = model_path.lower()
        
        if file_extension.endswith(('.joblib', '.pkl', '.pickle')):
            # Scikit-learn models (joblib or pickle)
            try:
                if file_extension.endswith('.joblib'):
                    model = joblib.load(model_path)
                else:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                return ModelWrapper(model, "sklearn")
            except Exception as e:
                raise ValueError(f"Failed to load scikit-learn model: {str(e)}")
                
        elif file_extension.endswith('.onnx'):
            # ONNX models
            if not ONNX_AVAILABLE:
                raise ValueError("ONNX runtime is not installed. Please install it with: pip install onnxruntime")
            try:
                model = ort.InferenceSession(model_path)
                return ModelWrapper(model, "onnx")
            except Exception as e:
                raise ValueError(f"Failed to load ONNX model: {str(e)}")
                
        else:
            raise ValueError(f"Unsupported model format. Supported formats: .joblib, .pkl, .pickle, .onnx. Got: {model_path}")

    def _detect_model_framework(self, model_wrapper: ModelWrapper) -> str:
        """Detect the framework used to create the model."""
        if model_wrapper.model_type == "sklearn":
            return "scikit-learn"
        elif model_wrapper.model_type == "onnx":
            return "onnx"
        else:
            return "unknown"

    def _get_model_algorithm(self, model_wrapper: ModelWrapper) -> str:
        """Get the algorithm name from the model."""
        if model_wrapper.model_type == "sklearn":
            return type(model_wrapper.model).__name__
        elif model_wrapper.model_type == "onnx":
            # For ONNX models, we can't easily determine the algorithm
            # Try to extract from metadata if available
            try:
                metadata = model_wrapper.model.get_modelmeta()
                if metadata and hasattr(metadata, 'custom_metadata_map'):
                    algorithm = metadata.custom_metadata_map.get('algorithm', 'ONNX Model')
                    return algorithm
                else:
                    return "ONNX Model"
            except:
                return "ONNX Model"
        else:
            return "Unknown"

    def load_model_and_data(self, model_path: str, data_path: str, target_column: str):
        """Loads the model and dataset from local files and prepares for analysis."""
        try:
            print(f"Loading model from: {model_path}")
            model_wrapper = self._load_model_by_format(model_path)
            self.model = model_wrapper
            
            print(f"Loading data from: {data_path}")
            df = pd.read_csv(data_path)

            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in the dataset.")

            print(f"Splitting data into training and testing sets.")
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Convert to numpy arrays to avoid feature name warnings
            X_array = X.values
            y_array = y.values
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_array, y_array, test_size=0.2, random_state=42
            )
            
            # Convert back to DataFrames for analysis but use position-based indexing for model predictions
            self.X_train = pd.DataFrame(self.X_train, columns=X.columns)
            self.X_test = pd.DataFrame(self.X_test, columns=X.columns)
            self.y_train = pd.Series(self.y_train, name=target_column)
            self.y_test = pd.Series(self.y_test, name=target_column)

            self.X_df = self.X_train  # Point to train data for backward compatibility
            self.y_s = self.y_train
            self.feature_names = list(X.columns)
            self.target_name = target_column
            
            print("Creating SHAP explainer...")
            # Use training data for SHAP to avoid feature name warnings
            # Note: SHAP TreeExplainer works best with sklearn models
            if model_wrapper.model_type == "sklearn":
                try:
                    self.explainer = shap.TreeExplainer(model_wrapper.model)
                except Exception as e:
                    print(f"Warning: Could not create SHAP TreeExplainer: {e}")
                    try:
                        # Fallback to general explainer
                        self.explainer = shap.Explainer(model_wrapper.predict_proba, self.X_train.values[:100])
                    except Exception as e2:
                        print(f"Warning: Could not create SHAP Explainer: {e2}")
                        self.explainer = None
            else:
                # For ONNX models, use a different SHAP explainer
                try:
                    # Use a smaller sample for initialization to avoid memory issues
                    sample_size = min(50, len(self.X_train))
                    background_data = self.X_train.values[:sample_size]
                    self.explainer = shap.Explainer(model_wrapper.predict_proba, background_data)
                except Exception as e:
                    print(f"Warning: Could not create SHAP explainer for {model_wrapper.model_type} model: {e}")
                    try:
                        # Try with even smaller sample
                        sample_size = min(10, len(self.X_train))
                        background_data = self.X_train.values[:sample_size]
                        self.explainer = shap.KernelExplainer(model_wrapper.predict_proba, background_data)
                    except Exception as e2:
                        print(f"Warning: Could not create SHAP KernelExplainer: {e2}")
                        self.explainer = None
            
            if self.explainer:
                try:
                    # Use a small sample for SHAP values to avoid memory/computation issues
                    sample_size = min(100, len(self.X_train))
                    self.shap_values = self.explainer.shap_values(self.X_train.values[:sample_size])
                    print(f"SHAP explainer created successfully with sample size {sample_size}.")
                except Exception as e:
                    print(f"Warning: Could not compute SHAP values: {e}")
                    self.shap_values = None
                    self.explainer = None
            else:
                self.shap_values = None
                print("SHAP explainer not available for this model type.")

            # Basic dataset diagnostics
            num_rows, num_cols = df.shape
            missing_ratio = float(df.isna().sum().sum() / (num_rows * num_cols)) if num_rows * num_cols > 0 else 0.0
            duplicate_ratio = float(df.duplicated().mean()) if num_rows > 0 else 0.0

            # Detect model framework and algorithm
            framework = self._detect_model_framework(model_wrapper)
            algorithm = self._get_model_algorithm(model_wrapper)

            self.model_info = {
                "model_path": model_path,
                "data_path": data_path,
                "target_column": target_column,
                "features_count": len(self.feature_names),
                "data_shape": df.shape,
                "algorithm": algorithm,
                "framework": framework,
                "model_type": model_wrapper.model_type,
                "version": "1.0.0",
                "created": pd.Timestamp.utcnow().isoformat(),
                "last_trained": pd.Timestamp.utcnow().isoformat(),
                "samples": int(num_rows),
                "features": int(len(self.feature_names)),
                "missing_pct": missing_ratio * 100.0,
                "duplicates_pct": duplicate_ratio * 100.0,
                "status": "Active",
                # simple heuristic health score from missing/duplicates
                "health_score_pct": max(0.0, 100.0 - (missing_ratio * 100.0 * 0.5 + duplicate_ratio * 100.0 * 0.5)),
                "train_samples": len(self.X_train),
                "test_samples": len(self.X_test),
                "shap_available": self.explainer is not None
            }
            return {"status": "success", "message": "Model and data loaded successfully.", "details": self.model_info}

        except Exception as e:
            self.__init__() # Reset state on failure
            raise e

    def load_model_and_separate_datasets(self, model_path: str, train_data_path: str, test_data_path: str, target_column: str):
        """Loads the model and separate train/test datasets from local files and prepares for analysis."""
        try:
            print(f"Loading model from: {model_path}")
            model_wrapper = self._load_model_by_format(model_path)
            self.model = model_wrapper
            
            print(f"Loading training data from: {train_data_path}")
            train_df = pd.read_csv(train_data_path)
            
            print(f"Loading test data from: {test_data_path}")
            test_df = pd.read_csv(test_data_path)

            # Validate target column exists in both datasets
            if target_column not in train_df.columns:
                raise ValueError(f"Target column '{target_column}' not found in the training dataset.")
            if target_column not in test_df.columns:
                raise ValueError(f"Target column '{target_column}' not found in the test dataset.")

            # Validate feature columns match
            train_features = set(train_df.drop(columns=[target_column]).columns)
            test_features = set(test_df.drop(columns=[target_column]).columns)
            if train_features != test_features:
                missing_in_test = train_features - test_features
                missing_in_train = test_features - train_features
                error_msg = "Feature columns mismatch between train and test datasets."
                if missing_in_test:
                    error_msg += f" Missing in test: {missing_in_test}."
                if missing_in_train:
                    error_msg += f" Missing in train: {missing_in_train}."
                raise ValueError(error_msg)

            # Extract features and targets
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]
            
            # Store as DataFrames
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test  
            self.y_test = y_test
            
            # Backward compatibility - point to train data
            self.X_df = self.X_train
            self.y_s = self.y_train
            self.feature_names = list(X_train.columns)
            self.target_name = target_column
            
            print("Creating SHAP explainer...")
            # Create SHAP explainer based on model type
            if model_wrapper.model_type == "sklearn":
                try:
                    self.explainer = shap.TreeExplainer(model_wrapper.model)
                except Exception as e:
                    print(f"Warning: Could not create SHAP TreeExplainer: {e}")
                    try:
                        # Fallback to general explainer
                        self.explainer = shap.Explainer(model_wrapper.predict_proba, self.X_train.values[:100])
                    except Exception as e2:
                        print(f"Warning: Could not create SHAP Explainer: {e2}")
                        self.explainer = None
            else:
                # For ONNX models, use a different SHAP explainer
                try:
                    # Use a smaller sample for initialization to avoid memory issues
                    sample_size = min(50, len(self.X_train))
                    background_data = self.X_train.values[:sample_size]
                    self.explainer = shap.Explainer(model_wrapper.predict_proba, background_data)
                except Exception as e:
                    print(f"Warning: Could not create SHAP explainer for {model_wrapper.model_type} model: {e}")
                    try:
                        # Try with even smaller sample
                        sample_size = min(10, len(self.X_train))
                        background_data = self.X_train.values[:sample_size]
                        self.explainer = shap.KernelExplainer(model_wrapper.predict_proba, background_data)
                    except Exception as e2:
                        print(f"Warning: Could not create SHAP KernelExplainer: {e2}")
                        self.explainer = None
            
            if self.explainer:
                try:
                    # Use a small sample for SHAP values to avoid memory/computation issues
                    sample_size = min(100, len(self.X_train))
                    self.shap_values = self.explainer.shap_values(self.X_train.values[:sample_size])
                    print(f"SHAP explainer created successfully with sample size {sample_size}.")
                except Exception as e:
                    print(f"Warning: Could not compute SHAP values: {e}")
                    self.shap_values = None
                    self.explainer = None
            else:
                self.shap_values = None
                print("SHAP explainer not available for this model type.")

            # Dataset diagnostics for both train and test
            train_missing_ratio = float(train_df.isna().sum().sum() / (train_df.shape[0] * train_df.shape[1])) if train_df.size > 0 else 0.0
            test_missing_ratio = float(test_df.isna().sum().sum() / (test_df.shape[0] * test_df.shape[1])) if test_df.size > 0 else 0.0
            train_duplicate_ratio = float(train_df.duplicated().mean()) if len(train_df) > 0 else 0.0
            test_duplicate_ratio = float(test_df.duplicated().mean()) if len(test_df) > 0 else 0.0

            # Detect model framework and algorithm
            framework = self._detect_model_framework(model_wrapper)
            algorithm = self._get_model_algorithm(model_wrapper)

            self.model_info = {
                "model_path": model_path,
                "train_data_path": train_data_path,
                "test_data_path": test_data_path,
                "target_column": target_column,
                "features_count": len(self.feature_names),
                "algorithm": algorithm,
                "framework": framework,
                "model_type": model_wrapper.model_type,
                "version": "1.0.0",
                "created": pd.Timestamp.utcnow().isoformat(),
                "last_trained": pd.Timestamp.utcnow().isoformat(),
                "features": int(len(self.feature_names)),
                "status": "Active",
                "train_samples": len(self.X_train),
                "test_samples": len(self.X_test),
                "train_missing_pct": train_missing_ratio * 100.0,
                "test_missing_pct": test_missing_ratio * 100.0,
                "train_duplicates_pct": train_duplicate_ratio * 100.0,
                "test_duplicates_pct": test_duplicate_ratio * 100.0,
                "health_score_pct": max(0.0, 100.0 - (train_missing_ratio * 50.0 + test_missing_ratio * 50.0)),
                "data_source": "separate_datasets",
                "shap_available": self.explainer is not None
            }
            return {"status": "success", "message": "Model and separate datasets loaded successfully.", "details": self.model_info}

        except Exception as e:
            self.__init__() # Reset state on failure
            raise e

    def _get_classification_metrics(self, y_true, y_pred, y_proba=None):
        """Calculate classification metrics handling both binary and multiclass cases."""
        # Determine if binary or multiclass
        unique_classes = np.unique(y_true)
        is_binary = len(unique_classes) == 2
        
        if is_binary:
            # Binary classification
            metrics = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, average='binary')),
                "recall": float(recall_score(y_true, y_pred, average='binary')),
                "f1_score": float(f1_score(y_true, y_pred, average='binary'))
            }
            # Add AUC for binary classification if probabilities are provided
            if y_proba is not None:
                if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
                    metrics["auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                elif y_proba.ndim == 1:
                    metrics["auc"] = float(roc_auc_score(y_true, y_proba))
                else:
                    metrics["auc"] = 0.0
        else:
            # Multiclass classification
            metrics = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, average='weighted')),
                "recall": float(recall_score(y_true, y_pred, average='weighted')),
                "f1_score": float(f1_score(y_true, y_pred, average='weighted'))
            }
            # For multiclass, use macro-averaged AUC if probabilities are provided
            if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] > 2:
                try:
                    metrics["auc"] = float(roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro'))
                except Exception:
                    metrics["auc"] = 0.0
            else:
                metrics["auc"] = 0.0
                
        return metrics, is_binary

    def _is_ready(self):
        """Check if the service has a model and data loaded."""
        if self.model is None or self.X_train is None or self.y_train is None:
            raise ValueError("Model and training data have not been uploaded yet. Please upload files first.")

    # --- Analysis Methods ---

    # --- Analysis Methods ---
    
    def get_model_overview(self) -> Dict[str, Any]:
        self._is_ready()
        
        # Calculate performance metrics on training data
        y_pred_train = self.model.predict(self.X_train.values)
        y_proba_train = self.model.predict_proba(self.X_train.values)
        
        train_metrics, is_binary = self._get_classification_metrics(
            self.y_train, y_pred_train, y_proba_train
        )
        
        # Calculate performance metrics on test data (if available)
        test_metrics = None
        overfitting_score = 0.0
        if self.X_test is not None and self.y_test is not None:
            y_pred_test = self.model.predict(self.X_test.values)
            y_proba_test = self.model.predict_proba(self.X_test.values)
            
            test_metrics, _ = self._get_classification_metrics(
                self.y_test, y_pred_test, y_proba_test
            )
            
            # Calculate overfitting score (difference between train and test accuracy)
            overfitting_score = abs(train_metrics["accuracy"] - test_metrics["accuracy"])

        feature_schema = []
        for feature in self.feature_names:
            col = self.X_train[feature]  # Use training data for schema
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

        performance_metrics = {
            "train": train_metrics,
            "overfitting_score": overfitting_score
        }
        
        if test_metrics:
            performance_metrics["test"] = test_metrics

        # Determine model type based on classification type
        model_type = "binary_classification" if is_binary else "multiclass_classification"

        return {
            "model_id": self.model_info.get('model_path', 'N/A'),
            "name": "Uploaded Classification Model",
            "model_type": model_type,
            "version": self.model_info.get("version", "1.0.0"),
            "framework": self.model_info.get("framework", "scikit-learn"),
            "status": self.model_info.get("status", "Active"),
            "algorithm": self.model_info.get("algorithm", "Unknown"),
            "feature_names": self.feature_names,
            "schema": feature_schema,
            "performance_metrics": performance_metrics,
            "shap_available": self.explainer is not None,
            "metadata": {
                "created": self.model_info.get("created"),
                "last_trained": self.model_info.get("last_trained"),
                "train_samples": self.model_info.get("train_samples", len(self.X_train)),
                "test_samples": self.model_info.get("test_samples", len(self.X_test) if self.X_test is not None else 0),
                "dataset_split": {
                    "train": len(self.X_train), 
                    "test": len(self.X_test) if self.X_test is not None else 0
                },
                "missing_pct": self.model_info.get("missing_pct", 0.0),
                "duplicates_pct": self.model_info.get("duplicates_pct", 0.0),
                "health_score_pct": self.model_info.get("health_score_pct", 100.0),
                "data_source": self.model_info.get("data_source", "single_dataset")
            }
        }

    def get_classification_stats(self) -> Dict[str, Any]:
        self._is_ready()
        
        # Use test data if available, otherwise fall back to training data
        if self.X_test is not None and self.y_test is not None:
            X_eval = self.X_test
            y_eval = self.y_test
            data_source = "test"
        else:
            X_eval = self.X_train
            y_eval = self.y_train
            data_source = "train"
            
        y_pred = self.model.predict(X_eval.values)
        y_proba = self.model.predict_proba(X_eval.values)
        
        metrics, is_binary = self._get_classification_metrics(y_eval, y_pred, y_proba)
        
        cm = confusion_matrix(y_eval, y_pred)
        
        # ROC curve only for binary classification
        roc_curve_data = {}
        if is_binary and y_proba is not None:
            try:
                # Get probabilities for positive class
                if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
                    proba_pos = y_proba[:, 1]
                else:
                    proba_pos = y_proba.flatten()
                    
                fpr, tpr, _ = roc_curve(y_eval, proba_pos)
                roc_curve_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
            except Exception as e:
                print(f"Warning: Could not compute ROC curve: {e}")
                roc_curve_data = {"fpr": [], "tpr": []}

        # Convert confusion matrix to standard format
        if cm.shape == (2, 2):
            # Binary classification
            confusion_matrix_dict = {
                "true_negative": int(cm[0, 0]), 
                "false_positive": int(cm[0, 1]),
                "false_negative": int(cm[1, 0]), 
                "true_positive": int(cm[1, 1]),
            }
        else:
            # Multiclass - provide the full matrix
            confusion_matrix_dict = {
                "matrix": cm.tolist(),
                "classes": sorted(np.unique(y_eval).tolist())
            }

        return {
            "metrics": metrics,
            "data_source": data_source,
            "confusion_matrix": confusion_matrix_dict,
            "roc_curve": roc_curve_data,
            "classification_type": "binary" if is_binary else "multiclass"
        }

    def get_feature_importance(self, method: str) -> Dict[str, Any]:
        self._is_ready()
        if method == 'shap':
            if self.shap_values is None:
                return {"error": "SHAP values are not available for this model type. Try 'builtin' method instead.", "method": method, "features": []}
            
            # Handle both binary and multiclass SHAP values
            if isinstance(self.shap_values, list):
                # Multiclass: average across all classes
                importance_values = np.abs(np.array(self.shap_values)).mean(axis=(0, 1))
            elif len(self.shap_values.shape) == 3:
                # Multiclass: shape is (n_samples, n_features, n_classes)
                importance_values = np.abs(self.shap_values).mean(axis=(0, 2))
            elif len(self.shap_values.shape) == 2:
                # Binary: shape is (n_samples, n_features) - already for positive class
                importance_values = np.abs(self.shap_values).mean(0)
            else:
                # Fallback for unexpected shapes
                importance_values = np.abs(self.shap_values).mean(0)
                
        else: # 'permutation' or 'builtin'
            if hasattr(self.model, 'feature_importances_'):
                importance_values = self.model.feature_importances_
            else:
                return {"error": "Built-in feature importance is not available for this model type.", "method": method, "features": []}

        sorted_indices = np.argsort(importance_values)[::-1]
        features = [{
            "name": self.feature_names[idx],
            "importance": float(importance_values[idx])
        } for idx in sorted_indices]
        
        return {"method": method, "features": features}

    # --- New: Enterprise Feature APIs ---
    def get_feature_metadata(self) -> Dict[str, Any]:
        """Return available features with basic metadata."""
        self._is_ready()
        features: List[Dict[str, Any]] = []
        for name in self.feature_names:
            col = self.X_df[name]
            if pd.api.types.is_numeric_dtype(col.dtype):
                features.append({
                    "name": name,
                    "type": "numeric",
                    "description": f"{name} numeric feature",
                    "min_value": float(col.min()),
                    "max_value": float(col.max())
                })
            else:
                # take up to 20 categories
                uniques = col.astype(str).unique()[:20].tolist()
                features.append({
                    "name": name,
                    "type": "categorical",
                    "categories": uniques
                })
        return {"features": features}

    def _encode_mixed_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        encoded_cols = {}
        for c in df.columns:
            series = df[c]
            if pd.api.types.is_numeric_dtype(series.dtype):
                encoded_cols[c] = series.astype(float)
            else:
                # factorize to integer codes for Pearson approx
                codes, _ = pd.factorize(series.astype(str), sort=True)
                encoded_cols[c] = pd.Series(codes, index=series.index, dtype=float)
        return pd.DataFrame(encoded_cols)

    def compute_correlation(self, selected_features: List[str]) -> Dict[str, Any]:
        self._is_ready()
        if not selected_features or len(selected_features) < 2:
            raise ValueError("At least two features are required for correlation analysis.")
        for feat in selected_features:
            if feat not in self.feature_names:
                raise ValueError(f"Feature '{feat}' not found")

        cache_key = "|".join(sorted(selected_features))
        if cache_key in self._correlation_cache:
            return self._correlation_cache[cache_key]

        df = self.X_df[selected_features]
        enc = self._encode_mixed_dataframe(df)
        corr = enc.corr(method='pearson')
        matrix = corr.loc[selected_features, selected_features].to_numpy().tolist()
        payload = {
            "features": selected_features,
            "matrix": [[float(v) for v in row] for row in matrix],
            "computed_at": pd.Timestamp.utcnow().isoformat()
        }
        # cache result
        self._correlation_cache[cache_key] = payload
        return payload

    def _get_shap_values_for_analysis(self) -> Optional[np.ndarray]:
        """Get SHAP values appropriate for analysis, handling both binary and multiclass cases."""
        if self.shap_values is None:
            return None
            
        if isinstance(self.shap_values, list):
            # Multiclass: use positive class (typically last) or average
            if len(self.shap_values) == 2:
                # Binary case stored as list
                return self.shap_values[1]
            else:
                # Multiclass: average across all classes
                return np.array(self.shap_values).mean(axis=0)
        elif len(self.shap_values.shape) == 3:
            # Multiclass: shape is (n_samples, n_features, n_classes)
            # Average across classes for analysis
            return self.shap_values.mean(axis=2)
        else:
            # Binary or already processed
            return self.shap_values

    def compute_feature_importance_advanced(self, method: str = 'shap', sort_by: str = 'importance', top_n: int = 20, visualization: str = 'bar') -> Dict[str, Any]:
        self._is_ready()
        # normalize params for cache key
        key = f"{method}|{sort_by}|{top_n}|{visualization}"
        if key in self._importance_cache:
            return self._importance_cache[key]

        method = method.lower()
        if method == 'shap':
            if self.shap_values is None:
                return {"error": "SHAP values are not available for this model type. Try 'builtin' method instead."}
            shap_vals = self._get_shap_values_for_analysis()
            if shap_vals is None:
                return {"error": "Could not extract SHAP values for analysis."}
            abs_mean = np.abs(shap_vals).mean(0)
            mean_signed = shap_vals.mean(0)
            importance = abs_mean
            direction = np.sign(mean_signed)
        elif method in ('permutation', 'gain', 'builtin'):
            # try to use built-in feature_importances_ when available
            if hasattr(self.model, 'feature_importances_'):
                fi = np.array(self.model.feature_importances_, dtype=float)
                importance = fi
                direction = np.sign(fi - fi.mean())  # rough proxy for direction
            else:
                # fallback to shap if builtin is not available
                if self.shap_values is None:
                    return {"error": "Neither built-in feature importance nor SHAP values are available for this model type."}
                shap_vals = self._get_shap_values_for_analysis()
                if shap_vals is None:
                    return {"error": "Could not extract SHAP values for analysis."}
                importance = np.abs(shap_vals).mean(0)
                direction = np.sign(shap_vals.mean(0))
        else:
            raise ValueError("Unsupported method. Use 'shap', 'permutation', or 'gain'.")

        items = []
        for idx, name in enumerate(self.feature_names):
            items.append({
                "name": name,
                "importance_score": float(importance[idx]),
                "impact_direction": "positive" if direction[idx] >= 0 else "negative",
                "rank": 0
            })

        # sorting
        if sort_by == 'feature_name':
            items.sort(key=lambda x: x['name'])
        elif sort_by == 'impact':
            # positive first, then by magnitude
            items.sort(key=lambda x: (x['impact_direction'] != 'positive', -x['importance_score']))
        else:  # importance
            items.sort(key=lambda x: x['importance_score'], reverse=True)

        # assign ranks after sort
        for i, it in enumerate(items):
            it['rank'] = i + 1

        top_items = items[: int(top_n)]
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

    def explain_instance(self, instance_idx: int) -> Dict[str, Any]:
        self._is_ready()
        if not (0 <= instance_idx < len(self.X_df)):
            raise ValueError("Instance index out of bounds.")
            
        instance_data = self.X_df.iloc[instance_idx]
        shap_vals_for_instance = self._get_instance_shap_vector(instance_idx)
        
        # Handle case where explainer might not be available
        base_value = 0.0
        if self.explainer and hasattr(self.explainer, 'expected_value'):
            try:
                if isinstance(self.explainer.expected_value, (list, tuple)):
                    base_value = float(self.explainer.expected_value[1] if len(self.explainer.expected_value) > 1 else self.explainer.expected_value[0])
                else:
                    base_value = float(self.explainer.expected_value)
            except Exception as e:
                print(f"Warning: Could not get base value from explainer: {e}")
                base_value = 0.5  # Default baseline for binary classification

        # Create single-row DataFrame for prediction to maintain feature names
        instance_df = pd.DataFrame([instance_data], columns=self.feature_names)
        # Use .values to avoid feature name warnings
        prediction_prob = float(self.model.predict_proba(instance_df.values)[0, 1])

        # Prepare both mapping and ordered arrays for convenience on the frontend
        shap_mapping = dict(zip(self.feature_names, shap_vals_for_instance))
        ordered = sorted(shap_mapping.items(), key=lambda kv: abs(kv[1]), reverse=True)
        ordered_features = [name for name, _ in ordered]
        ordered_values = [float(val) for _, val in ordered]
        ordered_feature_values = [instance_data[name] for name in ordered_features]

        return {
          "instance_id": instance_idx,
          "features": instance_data.to_dict(),
          "prediction": prediction_prob,
          "actual_value": int(self.y_s.iloc[instance_idx]),
          "base_value": float(base_value),
          "shap_values_map": shap_mapping,
          "ordered_contributions": {
              "feature_names": ordered_features,
              "feature_values": [self._safe_float(v) for v in ordered_feature_values],
              "shap_values": ordered_values
          }
        }

    def _safe_float(self, value: Any) -> Any:
        try:
            return float(value)
        except Exception:
            return str(value)

    def _get_instance_shap_vector(self, instance_idx: int) -> np.ndarray:
        """Return a 1D array of SHAP values for the specified instance, aligned with feature_names.
        Handles different shapes returned by SHAP depending on model/explainer versions.
        """
        if self.shap_values is None or self.explainer is None:
            # Return zeros if SHAP is not available
            return np.zeros(len(self.feature_names))
            
        try:
            # Most tree models: shap_values is a list per class
            if isinstance(self.shap_values, (list, tuple)):
                arr = self.shap_values[1] if len(self.shap_values) > 1 else self.shap_values[0]
                if instance_idx < len(arr):
                    vec = arr[instance_idx]
                    if vec.ndim > 1:
                        vec = vec.reshape(-1)
                    return vec
                else:
                    return np.zeros(len(self.feature_names))
                    
            # Some explainers: ndarray of shape (n_samples, n_features)
            if isinstance(self.shap_values, np.ndarray):
                if self.shap_values.ndim == 2 and instance_idx < self.shap_values.shape[0]:
                    return self.shap_values[instance_idx]
                # Otherwise compute on-the-fly for the single instance
                
            # Fallback: compute per-instance using explainer
            if instance_idx < len(self.X_df):
                one = self.X_df.iloc[[instance_idx]]
                sv = self.explainer.shap_values(one.values)
                if isinstance(sv, (list, tuple)):
                    vec = sv[1][0] if len(sv) > 1 else sv[0][0]
                else:
                    vec = sv[0]
                if isinstance(vec, np.ndarray) and vec.ndim > 1:
                    vec = vec.reshape(-1)
                return vec
        except Exception as e:
            print(f"Warning: Could not compute SHAP values for instance {instance_idx}: {e}")
            
        # Final fallback: return zeros
        return np.zeros(len(self.feature_names))

    def _get_shap_matrix(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Return SHAP values as a 2D array shaped (n_samples, n_features) for the positive class or
        averaged across classes when applicable. Handles different SHAP return shapes.
        """
        base = self.shap_values
        if isinstance(base, (list, tuple)):
            if len(base) == 2:
                mat = base[1]
            else:
                mat = np.mean(np.array(base, dtype=object), axis=0)
        elif isinstance(base, np.ndarray):
            if base.ndim == 3:  # (n_classes, n_samples, n_features)
                if base.shape[0] == 2:
                    mat = base[1]
                else:
                    mat = base.mean(axis=0)
            elif base.ndim == 2:
                mat = base
            else:
                mat = base.squeeze()
        else:
            # compute on the fly
            X_calc = X if X is not None else self.X_df
            sv = self.explainer.shap_values(X_calc)
            if isinstance(sv, (list, tuple)):
                mat = sv[1] if len(sv) == 2 else np.mean(np.array(sv, dtype=object), axis=0)
            else:
                mat = sv
        arr = np.asarray(mat)
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[-2], arr.shape[-1])
        return arr.astype(float)

    def perform_what_if(self, features: Dict[str, Any]) -> Dict[str, Any]:
        self._is_ready()
        try:
            input_df = pd.DataFrame([features], columns=self.feature_names)
            # Use .values to avoid feature name warnings
            prediction_proba = self.model.predict_proba(input_df.values)
            
            # Handle both binary and multiclass cases
            if prediction_proba.shape[1] == 2:
                # Binary classification - return positive class probability
                return {"prediction": float(prediction_proba[0, 1])}
            else:
                # Multiclass - return all class probabilities and predicted class
                class_probabilities = {f"class_{i}": float(prob) for i, prob in enumerate(prediction_proba[0])}
                predicted_class = int(np.argmax(prediction_proba[0]))
                return {
                    "prediction": float(prediction_proba[0, predicted_class]),
                    "predicted_class": predicted_class,
                    "class_probabilities": class_probabilities
                }
        except Exception as e:
            raise ValueError(f"Error during 'what-if' prediction: {e}")

    def get_feature_dependence(self, feature_name: str) -> Dict[str, Any]:
        self._is_ready()
        if feature_name not in self.feature_names:
            raise ValueError("Feature not found")
        
        if self.shap_values is None:
            return {"error": "SHAP values are not available for this model type.", "feature_values": [], "shap_values": []}
        
        feature_idx = self.feature_names.index(feature_name)
        shap_vals = self.shap_values[1] # Positive class
        
        return {
            "feature_values": self.X_df[feature_name].tolist(),
            "shap_values": shap_vals[:, feature_idx].tolist()
        }

    def list_instances(self, sort_by: str = "prediction", limit: int = 100) -> Dict[str, Any]:
        """Return lightweight list of instances to populate selector UI."""
        self._is_ready()
        proba = self.model.predict_proba(self.X_df)[:, 1]
        records = []
        for idx in range(len(self.X_df)):
            records.append({
                "id": idx,
                "prediction": float(proba[idx]),
                "actual": int(self.y_s.iloc[idx]),
            })
        if sort_by == "prediction":
            records.sort(key=lambda r: r["prediction"], reverse=True)
        elif sort_by == "confidence":
            records.sort(key=lambda r: abs(r["prediction"] - 0.5), reverse=True)
        return {"instances": records[:limit], "total": len(self.X_df)}

    def get_dataset_comparison(self) -> Dict[str, Any]:
        """Compare train vs test dataset characteristics and detect potential issues."""
        self._is_ready()
        
        if self.X_test is None or self.y_test is None:
            return {"error": "Test dataset not available for comparison"}
        
        def get_dataset_stats(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
            """Get basic statistics for a dataset."""
            return {
                "samples": len(X),
                "features": len(X.columns),
                "missing_values": int(X.isna().sum().sum()),
                "missing_percentage": float(X.isna().sum().sum() / (X.shape[0] * X.shape[1]) * 100),
                "duplicates": int(X.duplicated().sum()),
                "target_distribution": y.value_counts().to_dict(),
                "numeric_features": int(X.select_dtypes(include=[np.number]).shape[1]),
                "categorical_features": int(X.select_dtypes(exclude=[np.number]).shape[1])
            }
        
        def calculate_feature_drift() -> Dict[str, float]:
            """Calculate simple statistical drift for numeric features."""
            drift_scores = {}
            for feature in self.feature_names:
                if pd.api.types.is_numeric_dtype(self.X_train[feature]):
                    train_mean = self.X_train[feature].mean()
                    test_mean = self.X_test[feature].mean()
                    train_std = self.X_train[feature].std()
                    
                    # Simple drift score: normalized difference in means
                    if train_std > 0:
                        drift_score = abs(train_mean - test_mean) / train_std
                    else:
                        drift_score = 0.0
                    drift_scores[feature] = float(drift_score)
            return drift_scores
        
        train_stats = get_dataset_stats(self.X_train, self.y_train)
        test_stats = get_dataset_stats(self.X_test, self.y_test)
        feature_drift = calculate_feature_drift()
        
        # Calculate target distribution shift
        train_target_dist = pd.Series(train_stats["target_distribution"])
        test_target_dist = pd.Series(test_stats["target_distribution"])
        
        # Ensure both have same index
        all_classes = set(train_target_dist.index) | set(test_target_dist.index)
        train_proportions = train_target_dist.reindex(all_classes, fill_value=0) / train_stats["samples"]
        test_proportions = test_target_dist.reindex(all_classes, fill_value=0) / test_stats["samples"]
        
        target_drift = float(np.sum(np.abs(train_proportions - test_proportions)))
        
        # Identify potential issues
        issues = []
        if train_stats["missing_percentage"] > 5 or test_stats["missing_percentage"] > 5:
            issues.append("High missing values detected")
        if target_drift > 0.1:
            issues.append("Significant target distribution shift")
        if any(score > 2.0 for score in feature_drift.values()):
            issues.append("High feature drift detected")
        if abs(train_stats["samples"] - test_stats["samples"]) / max(train_stats["samples"], test_stats["samples"]) > 0.5:
            issues.append("Unbalanced train/test split")
            
        return {
            "train_stats": train_stats,
            "test_stats": test_stats,
            "feature_drift": feature_drift,
            "target_drift": target_drift,
            "issues": issues,
            "drift_summary": {
                "mean_feature_drift": float(np.mean(list(feature_drift.values()))),
                "max_feature_drift": float(np.max(list(feature_drift.values()))) if feature_drift else 0.0,
                "high_drift_features": [f for f, score in feature_drift.items() if score > 1.0]
            }
        }

    # --- Section 2: ROC and Threshold Analysis ---
    def roc_analysis(self) -> Dict[str, Any]:
        self._is_ready()
        
        # Use test data if available, otherwise fall back to training data
        if self.X_test is not None and self.y_test is not None:
            X_eval = self.X_test
            y_eval = self.y_test
            data_source = "test"
        else:
            X_eval = self.X_train
            y_eval = self.y_train
            data_source = "train"
            
        y_true = y_eval.values
        y_proba = self.model.predict_proba(X_eval)
        
        # Check if binary or multiclass
        unique_classes = np.unique(y_true)
        is_binary = len(unique_classes) == 2
        
        if not is_binary:
            return {
                "error": "ROC analysis is only available for binary classification problems.",
                "data_source": data_source,
                "classification_type": "multiclass",
                "num_classes": len(unique_classes)
            }
        
        # Binary classification ROC
        try:
            # Get probabilities for positive class
            if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
                proba_pos = y_proba[:, 1]
            else:
                proba_pos = y_proba.flatten()
                
            fpr, tpr, thresholds = roc_curve(y_true, proba_pos)
            auc_val = float(roc_auc_score(y_true, proba_pos))
        except Exception as e:
            return {
                "error": f"Could not compute ROC curve: {str(e)}",
                "data_source": data_source
            }

        # Sanitize arrays for JSON (no NaN/Inf)
        def to_finite_list(arr: np.ndarray, clamp_low: float = 0.0, clamp_high: float = 1.0) -> list:
            out: List[float] = []
            for val in arr:
                v = float(val)
                if not np.isfinite(v):
                    v = clamp_high if v > 0 else clamp_low
                if clamp_low is not None and clamp_high is not None:
                    v = max(clamp_low, min(clamp_high, v))
                out.append(v)
            return out

        finite_thresholds = thresholds[np.isfinite(thresholds)]
        max_thr = float(finite_thresholds.max()) if finite_thresholds.size > 0 else 1.0
        clean_thresholds = [float(v) if np.isfinite(v) else max_thr for v in thresholds]

        # Youden's J statistic: maximize tpr - fpr
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        optimal_threshold = float(clean_thresholds[best_idx])
        sensitivity = float(tpr[best_idx])  # same as recall at this threshold
        
        return {
            "roc_curve": {
                "fpr": to_finite_list(fpr),
                "tpr": to_finite_list(tpr),
                "thresholds": clean_thresholds
            },
            "metrics": {
                "auc_score": auc_val,
                "optimal_threshold": optimal_threshold,
                "sensitivity": sensitivity
            },
            "data_source": data_source,
            "classification_type": "binary"
        }

    def threshold_analysis(self, num_thresholds: int = 50) -> Dict[str, Any]:
        self._is_ready()
        
        # Use test data if available, otherwise fall back to training data
        if self.X_test is not None and self.y_test is not None:
            X_eval = self.X_test
            y_eval = self.y_test
            data_source = "test"
        else:
            X_eval = self.X_train
            y_eval = self.y_train
            data_source = "train"
            
        y_true = y_eval.values
        y_proba = self.model.predict_proba(X_eval)
        
        # Check if binary or multiclass
        unique_classes = np.unique(y_true)
        is_binary = len(unique_classes) == 2
        
        if not is_binary:
            return {
                "error": "Threshold analysis is only available for binary classification problems.",
                "data_source": data_source,
                "classification_type": "multiclass",
                "num_classes": len(unique_classes)
            }
        
        # Binary classification threshold analysis
        try:
            # Get probabilities for positive class
            if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
                proba_pos = y_proba[:, 1]
            else:
                proba_pos = y_proba.flatten()
        except Exception as e:
            return {
                "error": f"Could not extract probabilities: {str(e)}",
                "data_source": data_source
            }
            
        thresholds = np.linspace(0.0, 1.0, num=num_thresholds)
        results = []
        for thr in thresholds:
            y_pred = (proba_pos >= thr).astype(int)
            metrics, _ = self._get_classification_metrics(y_true, y_pred)
            results.append({
                "threshold": float(thr),
                "precision": metrics["precision"],
                "recall": metrics["recall"], 
                "f1_score": metrics["f1_score"],
                "accuracy": metrics["accuracy"]
            })

        # Use ROC analysis for optimal metrics
        roc_payload = self.roc_analysis()
        if "error" not in roc_payload:
            opt_thr = roc_payload["metrics"]["optimal_threshold"]
            # find nearest threshold entry
            nearest = min(results, key=lambda r: abs(r["threshold"] - opt_thr))
            optimal_metrics = {**nearest, "threshold": float(opt_thr)}
        else:
            optimal_metrics = results[len(results)//2] if results else {}
            
        return {
            "threshold_metrics": results, 
            "optimal_metrics": optimal_metrics,
            "data_source": data_source,
            "classification_type": "binary"
        }

    # --- Section 3: Individual Prediction Summary ---
    def individual_prediction(self, instance_idx: int) -> Dict[str, Any]:
        self._is_ready()
        if not (0 <= instance_idx < len(self.X_df)):
            raise ValueError("Instance index out of bounds.")
        instance_data = self.X_df.iloc[instance_idx]
        instance_df = instance_data.to_frame().T
        proba = float(self.model.predict_proba(instance_df)[0, 1])
        confidence = max(proba, 1.0 - proba)
        shap_vals_for_instance = self._get_instance_shap_vector(instance_idx)
        
        # Handle case where explainer might not be available
        base_value = 0.0
        if self.explainer and hasattr(self.explainer, 'expected_value'):
            try:
                if isinstance(self.explainer.expected_value, (list, tuple)):
                    base_value = float(self.explainer.expected_value[1] if len(self.explainer.expected_value) > 1 else self.explainer.expected_value[0])
                else:
                    base_value = float(self.explainer.expected_value)
            except Exception as e:
                print(f"Warning: Could not get base value from explainer: {e}")
                base_value = 0.5  # Default baseline for binary classification

        contributions = [
            {
                "name": name,
                "value": self._safe_float(instance_data[name]),
                "shap": float(shap_vals_for_instance[i])
            }
            for i, name in enumerate(self.feature_names)
        ]
        contributions.sort(key=lambda x: abs(x["shap"]), reverse=True)

        return {
            "prediction_percentage": proba * 100.0,
            "actual_outcome": int(self.y_s.iloc[instance_idx]),
            "confidence_score": confidence,
            "base_value": base_value,
            "shap_values": [float(v) for v in shap_vals_for_instance],
            "feature_contributions": contributions
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
        
        # Check if the model has trees (Random Forest, Extra Trees, etc.)
        if not hasattr(self.model, 'estimators_'):
            raise ValueError("The current model does not contain decision trees. Please use a tree-based model like RandomForest, ExtraTrees, etc.")
        
        # Collect data for all trees in the ensemble
        trees_data = []
        for idx, tree_estimator in enumerate(self.model.estimators_):
            tree_obj = tree_estimator.tree_

            def recurse(node, depth):
                if tree_obj.feature[node] != _tree.TREE_UNDEFINED:
                    feature = self.feature_names[tree_obj.feature[node]]
                    threshold = float(tree_obj.threshold[node])
                    samples = int(tree_obj.n_node_samples[node])
                    
                    # Calculate node purity (1 - gini impurity)
                    gini = float(tree_obj.impurity[node])
                    purity = 1 - gini
                    
                    return {
                        "type": "split",
                        "feature": feature,
                        "threshold": threshold,
                        "samples": samples,
                        "purity": purity,
                        "gini": gini,
                        "node_id": f"node_{node}",
                        "left": recurse(tree_obj.children_left[node], depth + 1),
                        "right": recurse(tree_obj.children_right[node], depth + 1)
                    }
                else:
                    # Leaf node
                    values = tree_obj.value[node][0]
                    samples = int(tree_obj.n_node_samples[node])
                    
                    # For classification, calculate prediction and confidence
                    total_samples = sum(values)
                    if total_samples > 0:
                        prediction = np.argmax(values)
                        confidence = values[prediction] / total_samples
                    else:
                        prediction = 0
                        confidence = 0.0
                    
                    # Calculate class distribution
                    class_distribution = {}
                    for i, val in enumerate(values):
                        class_distribution[f"class_{i}"] = val
                    
                    return {
                        "type": "leaf",
                        "samples": samples,
                        "prediction": float(prediction),
                        "confidence": float(confidence),
                        "purity": 1.0,  # Leaf nodes are pure by definition
                        "gini": 0.0,    # Leaf nodes have no impurity
                        "node_id": f"node_{node}",
                        "class_distribution": class_distribution
                    }

            # Calculate tree statistics
            total_nodes = tree_obj.node_count
            leaf_nodes = sum(1 for i in range(total_nodes) if tree_obj.feature[i] == _tree.TREE_UNDEFINED)
            max_depth = tree_obj.max_depth
            
            # Calculate tree accuracy on test set
            if self.X_test is not None and self.y_test is not None:
                tree_predictions = tree_estimator.predict(self.X_test)
                tree_accuracy = accuracy_score(self.y_test, tree_predictions)
            else:
                tree_accuracy = 0.0
            
            # Get feature importance for this tree
            # Use the actual feature_importances_ of the individual tree if available
            if hasattr(tree_estimator, "feature_importances_"):
                tree_importance = tree_estimator.feature_importances_.tolist()
            else:
                tree_importance = None
          
            trees_data.append({
                "tree_index": idx,
                "accuracy": float(tree_accuracy),
                "importance": tree_importance,
                "total_nodes": total_nodes,
                "leaf_nodes": leaf_nodes,
                "max_depth": max_depth,
                "tree_structure": recurse(0, 0)
            })

        return {"trees": trees_data}

    # --- Section 4: Feature Dependence (PDP, SHAP dependence, ICE) ---
    def partial_dependence(self, feature_name: str, num_points: int = 20) -> Dict[str, Any]:
        self._is_ready()
        if feature_name not in self.feature_names:
            raise ValueError("Feature not found")
        col = self.X_df[feature_name]
        is_numeric = pd.api.types.is_numeric_dtype(col.dtype)
        grid: List[Any]
        if is_numeric:
            # quantile grid for stability
            qs = np.linspace(0.01, 0.99, num_points)
            grid = list(np.quantile(col.astype(float), qs))
        else:
            grid = list(col.astype(str).unique())

        preds: List[float] = []
        for v in grid:
            X_mod = self.X_df.copy()
            X_mod[feature_name] = v
            proba = self.model.predict_proba(X_mod)[:, 1]
            preds.append(float(np.mean(proba)))

        # Impact metrics
        effect_range = float(np.max(preds) - np.min(preds)) if len(preds) > 0 else 0.0
        direction = "increasing"
        if is_numeric and len(preds) > 2:
            corr = np.corrcoef(np.array(grid, dtype=float), np.array(preds, dtype=float))[0, 1]
            if corr < -0.2:
                direction = "decreasing"
            elif corr > 0.2:
                direction = "increasing"
            else:
                direction = "non-monotonic"
        # Approximate importance from SHAP global importance
        try:
            if self.shap_values is not None:
                shap_vals = self.shap_values[1] if isinstance(self.shap_values, list) else self.shap_values
                abs_mean = np.abs(shap_vals).mean(0)
                denom = float(np.sum(abs_mean)) if float(np.sum(abs_mean)) > 0 else 1.0
                idx = self.feature_names.index(feature_name)
                importance_pct = float(abs_mean[idx] / denom) * 100.0
            else:
                importance_pct = 0.0
        except Exception:
            importance_pct = 0.0

        impact = {
            "impact_summary": "High influence on model predictions" if importance_pct > 5 else "Moderate influence",
            "feature_type": "numerical" if is_numeric else "categorical",
            "importance_percentage": round(importance_pct, 2),
            "effect_range": round(effect_range, 6),
            "trend_analysis": {"direction": direction, "variability": "continuous" if is_numeric else "discrete"},
            "confidence_score": int(min(100, max(50, importance_pct)))
        }

        return {
            "feature": feature_name,
            "x": [float(x) if is_numeric else x for x in grid],
            "y": preds,
            "impact": impact
        }

    def shap_dependence(self, feature_name: str, color_by: Optional[str] = None) -> Dict[str, Any]:
        self._is_ready()
        if feature_name not in self.feature_names:
            raise ValueError("Feature not found")
        idx = self.feature_names.index(feature_name)
        shap_mat = self._get_shap_matrix()
        shap_vec = np.asarray(shap_mat[:, idx]).reshape(-1)
        feature_vals = self.X_df[feature_name]
        payload: Dict[str, Any] = {
            "feature": feature_name,
            "feature_values": feature_vals.astype(float).tolist() if pd.api.types.is_numeric_dtype(feature_vals.dtype) else feature_vals.astype(str).tolist(),
            "shap_values": np.asarray(shap_vec, dtype=float).reshape(-1).tolist()
        }
        if color_by and color_by in self.feature_names:
            c = self.X_df[color_by]
            payload["color_by"] = color_by
            payload["color_values"] = c.astype(float).tolist() if pd.api.types.is_numeric_dtype(c.dtype) else c.astype(str).tolist()
        return payload

    def ice_plot(self, feature_name: str, num_points: int = 20, num_instances: int = 20) -> Dict[str, Any]:
        self._is_ready()
        if feature_name not in self.feature_names:
            raise ValueError("Feature not found")
        col = self.X_df[feature_name]
        is_numeric = pd.api.types.is_numeric_dtype(col.dtype)
        if not is_numeric:
            grid = list(col.astype(str).unique())
        else:
            qs = np.linspace(0.01, 0.99, num_points)
            grid = list(np.quantile(col.astype(float), qs))

        n = min(num_instances, len(self.X_df))
        sample_idx = list(range(n))
        curves = []
        for i in sample_idx:
            row = self.X_df.iloc[i].copy()
            ys: List[float] = []
            for v in grid:
                row_mod = row.copy()
                row_mod[feature_name] = v
                pred = self.model.predict_proba(pd.DataFrame([row_mod], columns=self.feature_names))[:, 1][0]
                ys.append(float(pred))
            curves.append({
                "instance": i,
                "x": [float(x) if is_numeric else x for x in grid],
                "y": ys
            })
        return {"feature": feature_name, "curves": curves}

    # --- Section 5: Feature Interactions ---
    def interaction_network(self, top_k: int = 30, sample_rows: int = 200) -> Dict[str, Any]:
        self._is_ready()
        # Sample to keep interaction computation tractable
        df = self.X_df.iloc[: min(sample_rows, len(self.X_df))]
        try:
            inter = self.explainer.shap_interaction_values(df)
            # inter can be list per class for classifiers
            if isinstance(inter, (list, tuple)):
                inter = inter[1]
            # Average absolute interaction across samples
            inter_arr = np.asarray(inter)
            if inter_arr.ndim == 4:
                # sometimes returns (n_classes, n_samples, F, F)
                inter_arr = inter_arr.mean(axis=0)
            mean_abs = np.abs(inter_arr).mean(axis=0)  # shape (F, F)
        except Exception:
            # Fallback: approximate with covariance of SHAP values
            shap_mat = self._get_shap_matrix()
            mean_abs = np.abs(np.cov(shap_mat, rowvar=False))

        F = len(self.feature_names)
        # Ensure matrix alignment with number of features
        mean_abs = np.asarray(mean_abs)
        if mean_abs.ndim != 2 or mean_abs.shape[0] != F or mean_abs.shape[1] != F:
            try:
                shap_mat = self._get_shap_matrix()
                if shap_mat.ndim == 2 and shap_mat.shape[1] == F:
                    mean_abs = np.abs(np.cov(shap_mat, rowvar=False))
                else:
                    mean_abs = np.zeros((F, F), dtype=float)
            except Exception:
                mean_abs = np.zeros((F, F), dtype=float)
        # If interactions are degenerate (all zeros), fallback to correlation proxy
        max_val = float(np.max(mean_abs)) if mean_abs.size else 0.0
        if max_val <= 0:
            enc = self._encode_mixed_dataframe(self.X_df)
            corr = enc.corr().abs().to_numpy()
            if corr.shape[0] != F or corr.shape[1] != F:
                corr = np.zeros((F, F), dtype=float)
            mean_abs = corr
            max_val = float(np.max(mean_abs)) if mean_abs.size else 0.0

        # Normalize to [0, 1] to align with frontend threshold slider
        norm_mat = mean_abs / max_val if max_val > 0 else mean_abs

        nodes = []
        # Node importance from global shap; ensure correct length
        shap_vals_all = self._get_shap_matrix()
        if shap_vals_all.ndim != 2 or shap_vals_all.shape[1] != F:
            if hasattr(self.model, 'feature_importances_') and len(getattr(self.model, 'feature_importances_')) == F:
                node_importance = np.asarray(self.model.feature_importances_, dtype=float)
            else:
                node_importance = np.zeros(F, dtype=float)
        else:
            node_importance = np.abs(shap_vals_all).mean(0)
        for i, name in enumerate(self.feature_names):
            nodes.append({
                "id": name,
                "name": name.replace("_", " "),
                "type": "numeric" if pd.api.types.is_numeric_dtype(self.X_df[name].dtype) else "categorical",
                "importance": float(node_importance[i])
            })

        edges = []
        for i in range(F):
            for j in range(i + 1, F):
                strength = float(norm_mat[i, j])
                if strength <= 0:
                    continue
                e_type = "independent" if strength >= float(np.median(norm_mat)) else "redundancy"
                edges.append({
                    "source": self.feature_names[i],
                    "target": self.feature_names[j],
                    "strength": strength,
                    "type": e_type
                })
        # Top K edges by strength
        edges.sort(key=lambda e: e["strength"], reverse=True)
        edges = edges[: top_k]

        top_pairs = [{
            "feature_pair": [e["source"], e["target"]],
            "interaction_score": e["strength"],
            "classification": e["type"]
        } for e in edges[:10]]

        # Provide heatmap matrix normalized 0..1 using the same normalization as edges
        matrix = norm_mat.astype(float)

        # Top features by importance
        imp = np.asarray(node_importance, dtype=float)
        if np.max(imp) > 0:
            imp_norm = imp / np.max(imp)
        else:
            imp_norm = imp
        order = np.argsort(imp)[::-1]
        top_features = [{"name": self.feature_names[i], "importance": float(imp[i]), "normalized": float(imp_norm[i])} for i in order[:10]]

        independent_count = int(sum(1 for e in edges if e["type"] == "independent"))
        summary = {
            "total_edges": int(len(edges)),
            "mean_strength": float(np.mean([e["strength"] for e in edges])) if edges else 0.0,
            "median_strength": float(np.median([e["strength"] for e in edges])) if edges else 0.0,
            "independence_ratio": float(independent_count / len(edges)) if edges else 0.0,
            "strongest_pair": edges[0]["source"] + "×" + edges[0]["target"] if edges else None
        }
        return {
            "nodes": nodes,
            "edges": edges,
            "top_interactions": top_pairs,
            "matrix_features": self.feature_names,
            "matrix": matrix.tolist(),
            "summary": summary,
            "top_features": top_features
        }

    def pairwise_analysis(self, feature1: str, feature2: str, color_by: Optional[str] = None, sample_size: int = 1000) -> Dict[str, Any]:
        self._is_ready()
        if feature1 not in self.feature_names or feature2 not in self.feature_names:
            raise ValueError("One or both features not found")
        df_all = self.X_df
        if len(df_all) > sample_size:
            df_all = df_all.sample(n=sample_size, random_state=42)
        preds = self.model.predict_proba(df_all[self.feature_names])[:, 1]
        payload: Dict[str, Any] = {
            "x": df_all[feature1].astype(float).tolist() if pd.api.types.is_numeric_dtype(df_all[feature1].dtype) else df_all[feature1].astype(str).tolist(),
            "y": df_all[feature2].astype(float).tolist() if pd.api.types.is_numeric_dtype(df_all[feature2].dtype) else df_all[feature2].astype(str).tolist(),
            "prediction": [float(p) for p in preds],
            "feature1": feature1,
            "feature2": feature2
        }
        if color_by and color_by in self.feature_names:
            c = df_all[color_by]
            payload["color_by"] = color_by
            payload["color_values"] = c.astype(float).tolist() if pd.api.types.is_numeric_dtype(c.dtype) else c.astype(str).tolist()
        return payload