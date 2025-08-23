import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import shap
import joblib
import pickle
from typing import Dict, Any, List, Optional

# We only support scikit-learn models to preserve tree structure for explainability

class ModelWrapper:
    """Wrapper class to provide consistent interface for scikit-learn models."""
    
    def __init__(self, model, model_type: str):
        self.model = model
        self.model_type = model_type
        self._feature_names_expected = None  # Will be determined during first prediction
        
    def _detect_feature_names_requirement(self, X):
        """
        Detect whether the model expects feature names or not by testing both approaches.
        This is done only once and cached for performance.
        """
        if self._feature_names_expected is not None:
            return self._feature_names_expected
            
        if not hasattr(X, 'values'):  # Already numpy array
            self._feature_names_expected = False
            return False
            
        # Test both approaches to see which one works without warnings
        import warnings
        
        # Test with DataFrame (feature names)
        with warnings.catch_warnings(record=True) as w1:
            warnings.simplefilter("always")
            try:
                _ = self.model.predict_proba(X.iloc[:1])
                df_warnings = len(w1)
            except:
                df_warnings = float('inf')  # Failed completely
        
        # Test with numpy array (no feature names)
        with warnings.catch_warnings(record=True) as w2:
            warnings.simplefilter("always")
            try:
                _ = self.model.predict_proba(X.iloc[:1].values)
                array_warnings = len(w2)
            except:
                array_warnings = float('inf')  # Failed completely
        
        # Choose the approach with fewer warnings
        if df_warnings <= array_warnings:
            self._feature_names_expected = True
            print(f"📊 Model expects feature names (DataFrame input) - will use pandas DataFrames for predictions")
        else:
            self._feature_names_expected = False
            print(f"📊 Model expects numpy arrays (no feature names) - will use .values for predictions")
            
        return self._feature_names_expected
    
    def _prepare_input(self, X):
        """Prepare input data in the format expected by the model."""
        if not hasattr(X, 'values'):  # Already numpy array
            return X
            
        expects_feature_names = self._detect_feature_names_requirement(X)
        
        if expects_feature_names:
            return X  # Return DataFrame as-is
        else:
            return X.values  # Convert to numpy array
        
    def predict(self, X):
        """Predict class labels with adaptive input format."""
        if self.model_type == "sklearn":
            prepared_X = self._prepare_input(X)
            return self.model.predict(prepared_X)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def predict_proba(self, X):
        """Predict class probabilities with adaptive input format."""
        if self.model_type == "sklearn":
            prepared_X = self._prepare_input(X)
            return self.model.predict_proba(prepared_X)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def __getattr__(self, name):
        """Delegate attribute access to the wrapped model."""
        return getattr(self.model, name)


class BaseModelService:
    """
    Base service that holds the loaded model and shared data/state.
    Other services will inherit from this or receive an instance.
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
        
        print("BaseModelService initialized. Waiting for model and data.")

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
                raise ValueError(f"Failed to load sklearn model from {model_path}: {str(e)}")
        else:
            raise ValueError(f"Unsupported model format. Expected .joblib, .pkl, or .pickle, got {file_extension}")

    def _detect_model_framework(self, model_wrapper: ModelWrapper) -> str:
        """Detect the framework used to create the model."""
        if model_wrapper.model_type == "sklearn":
            return "scikit-learn"
        else:
            return "unknown"

    def _get_model_algorithm(self, model_wrapper: ModelWrapper) -> str:
        """Get the algorithm name from the model."""
        if model_wrapper.model_type == "sklearn":
            model_name = type(model_wrapper.model).__name__
            # Handle ensemble models
            if hasattr(model_wrapper.model, 'base_estimator'):
                base_name = type(model_wrapper.model.base_estimator).__name__
                return f"{model_name}({base_name})"
            elif hasattr(model_wrapper.model, 'estimators_'):
                if hasattr(model_wrapper.model, 'base_estimator_'):
                    base_name = type(model_wrapper.model.base_estimator_).__name__
                    return f"{model_name}({base_name})"
            return model_name
        else:
            return "Unknown"

    def load_model_and_datasets(self, model_path: str, data_path: str = None, train_data_path: str = None, test_data_path: str = None, target_column: Optional[str] = None):
        """
        Unified method to load model and dataset(s) from local files. 
        Supports both single dataset and separate train/test scenarios.
        
        Args:
            model_path: Path to the model file
            data_path: Path to single dataset file (optional)
            train_data_path: Path to training dataset file (optional)
            test_data_path: Path to test dataset file (optional)
            target_column: Name of the target column
            
        Returns:
            Dict with loading results and model info
        """
        try:
            # Validate input parameters
            if data_path and (train_data_path or test_data_path):
                raise ValueError("Provide either data_path OR train_data_path+test_data_path, not both")
            
            if not data_path and not (train_data_path and test_data_path):
                raise ValueError("Must provide either data_path OR both train_data_path and test_data_path")
            
            if not target_column:
                raise ValueError("Target column must be specified")
            
            # Load model
            model_wrapper = self._load_model_by_format(model_path)
            self.model = model_wrapper
            
            if data_path:
                # Single dataset scenario
                return self._load_single_dataset(model_wrapper, model_path, data_path, target_column)
            else:
                # Separate train/test datasets scenario
                return self._load_separate_datasets(model_wrapper, model_path, train_data_path, test_data_path, target_column)
                
        except Exception as e:
            raise ValueError(f"Failed to load model and datasets: {str(e)}")
    
    def _load_single_dataset(self, model_wrapper: ModelWrapper, model_path: str, data_path: str, target_column: str):
        """Load model with single dataset and split into train/test."""
        # Load data
        if not data_path.endswith('.csv'):
            raise ValueError("Only CSV data files are supported.")
        
        df = pd.read_csv(data_path)
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")
        
        # Store metadata
        self.model_info = {
            'model_path': model_path,
            'data_path': data_path,
            'target_column': target_column,
            'framework': self._detect_model_framework(model_wrapper),
            'algorithm': self._get_model_algorithm(model_wrapper),
            'version': '1.0.0',
            'status': 'Active',
            'created': pd.Timestamp.utcnow().isoformat(),
            'last_trained': pd.Timestamp.utcnow().isoformat(),
            'data_source': 'single_dataset'
        }
        
        # Split features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Store feature information
        self.feature_names = X.columns.tolist()
        self.target_name = target_column
        
        # Split into train/test (80/20 split)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Store datasets
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        # For backward compatibility, point to training data
        self.X_df = X_train
        self.y_s = y_train
        
        # Update metadata with split information
        self.model_info.update({
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'missing_pct': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
            'duplicates_pct': (df.duplicated().sum() / len(df)) * 100,
            'health_score_pct': 100.0 - self.model_info.get('missing_pct', 0) - self.model_info.get('duplicates_pct', 0)
        })
        
        # Initialize SHAP explainer
        self._initialize_shap_explainer(model_wrapper, X_train)
        
        return {
            "message": "Model and data loaded successfully",
            "model_info": {
                "framework": self.model_info['framework'],
                "algorithm": self.model_info['algorithm'],
                "features": len(self.feature_names),
                "samples": len(df),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "target_column": target_column,
                "shap_available": self.explainer is not None
            }
        }
    
    def _load_separate_datasets(self, model_wrapper: ModelWrapper, model_path: str, train_data_path: str, test_data_path: str, target_column: str):
        """Load model with separate train and test datasets."""
        # Load training data
        if not train_data_path.endswith('.csv'):
            raise ValueError("Only CSV data files are supported.")
        
        train_df = pd.read_csv(train_data_path)
        
        if target_column not in train_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in training dataset.")
        
        # Load test data
        if not test_data_path.endswith('.csv'):
            raise ValueError("Only CSV data files are supported.")
        
        test_df = pd.read_csv(test_data_path)
        
        if target_column not in test_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in test dataset.")
        
        # Store metadata
        self.model_info = {
            'model_path': model_path,
            'train_data_path': train_data_path,
            'test_data_path': test_data_path,
            'target_column': target_column,
            'framework': self._detect_model_framework(model_wrapper),
            'algorithm': self._get_model_algorithm(model_wrapper),
            'version': '1.0.0',
            'status': 'Active',
            'created': pd.Timestamp.utcnow().isoformat(),
            'last_trained': pd.Timestamp.utcnow().isoformat(),
            'data_source': 'separate_datasets'
        }
        
        # Split features and target for training data
        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]
        
        # Split features and target for test data
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]
        
        # Ensure feature consistency between train and test
        if set(X_train.columns) != set(X_test.columns):
            raise ValueError("Training and test datasets have different features.")
        
        # Reorder test features to match training order
        X_test = X_test[X_train.columns]
        
        # Store feature information
        self.feature_names = X_train.columns.tolist()
        self.target_name = target_column
        
        # Store datasets
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        # For backward compatibility, point to training data
        self.X_df = X_train
        self.y_s = y_train
        
        # Update metadata with dataset information
        missing_train_pct = (train_df.isnull().sum().sum() / (train_df.shape[0] * train_df.shape[1])) * 100
        missing_test_pct = (test_df.isnull().sum().sum() / (test_df.shape[0] * test_df.shape[1])) * 100
        duplicates_train_pct = (train_df.duplicated().sum() / len(train_df)) * 100
        duplicates_test_pct = (test_df.duplicated().sum() / len(test_df)) * 100
        
        self.model_info.update({
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'missing_pct': (missing_train_pct + missing_test_pct) / 2,
            'duplicates_pct': (duplicates_train_pct + duplicates_test_pct) / 2,
            'health_score_pct': 100.0 - ((missing_train_pct + missing_test_pct) / 2) - ((duplicates_train_pct + duplicates_test_pct) / 2)
        })
        
        # Initialize SHAP explainer
        self._initialize_shap_explainer(model_wrapper, X_train)
        
        return {
            "message": "Model and separate datasets loaded successfully",
            "model_info": {
                "framework": self.model_info['framework'],
                "algorithm": self.model_info['algorithm'],
                "features": len(self.feature_names),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "target_column": target_column,
                "shap_available": self.explainer is not None
            }
        }
    
    def _initialize_shap_explainer(self, model_wrapper: ModelWrapper, X_train: pd.DataFrame):
        """Initialize SHAP explainer for tree-based models."""
        try:
            if model_wrapper.model_type == "sklearn" and (hasattr(model_wrapper.model, 'tree_') or hasattr(model_wrapper.model, 'estimators_')):
                self.explainer = shap.TreeExplainer(model_wrapper.model)
                # Compute SHAP values on a sample for efficiency
                sample_size = min(1000, len(X_train))
                sample_X = X_train.sample(n=sample_size, random_state=42)
                self.shap_values = self.explainer.shap_values(sample_X)
                print(f"SHAP explainer initialized with {sample_size} samples.")
            else:
                print("SHAP explainer not initialized (model type not supported or not tree-based)")
        except Exception as e:
            print(f"Failed to initialize SHAP explainer: {e}")
            self.explainer = None
            self.shap_values = None

    # Backward compatibility wrapper methods
    def load_model_and_data(self, model_path: str, data_path: str, target_column: str):
        """Legacy method for single dataset loading."""
        return self.load_model_and_datasets(model_path, data_path=data_path, target_column=target_column)

    def load_model_and_separate_datasets(self, model_path: str, train_data_path: str, test_data_path: str, target_column: str):
        """Legacy method for separate datasets loading."""
        return self.load_model_and_datasets(model_path, train_data_path=train_data_path, test_data_path=test_data_path, target_column=target_column)

    def _get_classification_metrics(self, y_true, y_pred, y_proba=None):
        """Calculate classification metrics handling both binary and multiclass cases."""
        # Determine if binary or multiclass
        unique_classes = np.unique(y_true)
        is_binary = len(unique_classes) == 2
        
        if is_binary:
            # Binary classification metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='binary')
            recall = recall_score(y_true, y_pred, average='binary')
            f1 = f1_score(y_true, y_pred, average='binary')
            
            # ROC AUC only if probabilities are available
            roc_auc = None
            if y_proba is not None:
                try:
                    # For binary classification, use positive class probabilities
                    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                        roc_auc = roc_auc_score(y_true, y_proba[:, 1])
                    else:
                        roc_auc = roc_auc_score(y_true, y_proba.flatten())
                except Exception:
                    roc_auc = None
                    
        else:
            # Multiclass metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            
            # Multiclass ROC AUC
            roc_auc = None
            if y_proba is not None:
                try:
                    roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                except Exception:
                    roc_auc = None
                    
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc) if roc_auc is not None else None
        }, is_binary

    def safe_predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Safe prediction method that automatically handles feature name compatibility.
        Use this method instead of calling model.predict() directly.
        """
        self._is_ready()
        return self.model.predict(X)
    
    def safe_predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Safe prediction probability method that automatically handles feature name compatibility.
        Use this method instead of calling model.predict_proba() directly.
        """
        self._is_ready()
        return self.model.predict_proba(X)
    
    def get_model_input_format(self) -> str:
        """
        Get information about what input format the model expects.
        Returns: 'dataframe' if model expects feature names, 'array' if it expects numpy arrays.
        """
        if self.model is None:
            return "unknown"
        
        if hasattr(self.model, '_feature_names_expected'):
            if self.model._feature_names_expected is True:
                return "dataframe"
            elif self.model._feature_names_expected is False:
                return "array"
        
        return "unknown"

    def _is_ready(self):
        """Check if the service has a model and data loaded."""
        if self.model is None or self.X_train is None or self.y_train is None:
            raise ValueError("Model and data must be loaded before performing analysis. Use the upload endpoints first.")

    def _safe_float(self, value: Any) -> Any:
        """Convert value to float safely, handling non-numeric values."""
        try:
            return float(value)
        except Exception:
            return value

    def _get_shap_values_for_analysis(self) -> Optional[np.ndarray]:
        """Get SHAP values appropriate for analysis, handling both binary and multiclass cases."""
        if self.shap_values is None:
            return None
            
        # Handle different SHAP value formats
        if isinstance(self.shap_values, list):
            # Multi-class case: SHAP returns list of arrays, one per class
            # For binary classification, take the positive class (index 1)
            if len(self.shap_values) == 2:
                return self.shap_values[1]  # Positive class
            else:
                # Multi-class: average across classes or take first class
                return self.shap_values[0]
        elif len(self.shap_values.shape) == 3:
            # 3D array: (n_samples, n_features, n_classes)
            # For binary, take positive class; for multi-class, take first class
            if self.shap_values.shape[2] == 2:
                return self.shap_values[:, :, 1]  # Positive class
            else:
                return self.shap_values[:, :, 0]  # First class
        else:
            # 2D array: (n_samples, n_features) - already in correct format
            return self.shap_values

    def _get_shap_matrix(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Return SHAP values as a 2D array shaped (n_samples, n_features) for the positive class or
        averaged across classes when applicable. Handles different SHAP return shapes.
        """
        base = self.shap_values
        if isinstance(base, (list, tuple)):
            # Multi-class: list of arrays
            if len(base) >= 2:
                mat = base[1]  # Positive class for binary
            else:
                mat = base[0]  # First class
        elif isinstance(base, np.ndarray):
            if base.ndim == 3:
                # 3D: take positive class or first class
                if base.shape[2] >= 2:
                    mat = base[:, :, 1]  # Positive class
                else:
                    mat = base[:, :, 0]  # First class
            else:
                mat = base  # Already 2D
        else:
            # Fallback: create zeros
            mat = np.zeros((len(self.X_df), len(self.feature_names)))
        
        arr = np.asarray(mat)
        if arr.ndim > 2:
            # Additional safeguard: flatten extra dimensions
            arr = arr.reshape(arr.shape[0], -1)[:, :len(self.feature_names)]
        return arr.astype(float)

    def _get_instance_shap_vector(self, instance_idx: int) -> np.ndarray:
        """Return a 1D array of SHAP values for the specified instance, aligned with feature_names.
        Handles different shapes returned by SHAP depending on model/explainer versions.
        """
        if self.shap_values is None or self.explainer is None:
            return np.zeros(len(self.feature_names))
            
        try:
            # Get the SHAP matrix and extract the instance
            shap_matrix = self._get_shap_matrix()
            if instance_idx < len(shap_matrix):
                return shap_matrix[instance_idx]
            else:
                return np.zeros(len(self.feature_names))
        except Exception as e:
            print(f"Error getting SHAP values for instance {instance_idx}: {e}")
            return np.zeros(len(self.feature_names))
