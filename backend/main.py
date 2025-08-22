from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List

from app.core.config import settings
from app.core.auth import verify_token
from app.services.model_service import ModelService
from app.services.ai_explanation_service import AIExplanationService
from app.services.s3_service import S3Service

app = FastAPI(title=settings.PROJECT_NAME, version=settings.PROJECT_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_service = ModelService()
ai_explanation_service = AIExplanationService()
s3_service = S3Service()


# Auto-load functionality disabled as requested by user
# User will upload model and dataset through the frontend interface

# --- Utility Function for Error Handling ---
def handle_request(service_func, *args, **kwargs):
    try:
        result = service_func(*args, **kwargs)
        return JSONResponse(status_code=200, content=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # For debugging, print the full error
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

# --- API Endpoints ---

@app.get("/", tags=["Root"])
def read_root():
    return {"message": f"Welcome to the {settings.PROJECT_NAME}"}

@app.post("/load/model-and-data", tags=["Setup"])
async def load_model_and_data_from_s3(
    payload: Dict = Body(...),
    token: str = Depends(verify_token)
):
    """
    Unified endpoint to load model and dataset(s) from S3 bucket. 
    Supports both single dataset and separate train/test scenarios.
    
    Expected payload for single dataset:
    {
        "model_s3_key": "path/to/model.joblib",
        "data_s3_key": "path/to/dataset.csv",
        "target_column": "target_variable_name"
    }
    
    Expected payload for separate train/test datasets:
    {
        "model_s3_key": "path/to/model.joblib",
        "train_s3_key": "path/to/train_dataset.csv",
        "test_s3_key": "path/to/test_dataset.csv",
        "target_column": "target_variable_name"
    }
    """
    try:
        model_s3_key = payload.get("model_s3_key")
        data_s3_key = payload.get("data_s3_key")
        train_s3_key = payload.get("train_s3_key")
        test_s3_key = payload.get("test_s3_key")
        target_column = payload.get("target_column")
        
        if not model_s3_key:
            raise HTTPException(status_code=400, detail="Missing 'model_s3_key' in payload")
        if not target_column:
            raise HTTPException(status_code=400, detail="Missing 'target_column' in payload")
        
        # Validate input scenarios
        if data_s3_key and (train_s3_key or test_s3_key):
            raise HTTPException(status_code=400, detail="Provide either 'data_s3_key' OR 'train_s3_key'+'test_s3_key', not both")
        
        if not data_s3_key and not (train_s3_key and test_s3_key):
            raise HTTPException(status_code=400, detail="Must provide either 'data_s3_key' OR both 'train_s3_key' and 'test_s3_key'")
        
        # Download files from S3 using unified function
        download_result = s3_service.download_model_and_datasets(
            model_s3_key=model_s3_key,
            data_s3_key=data_s3_key,
            train_s3_key=train_s3_key,
            test_s3_key=test_s3_key
        )
        
        # Check if download was successful
        if not download_result or download_result[0] is None:
            raise HTTPException(status_code=500, detail="Failed to download files from S3")
        
        # Use unified service method
        if data_s3_key:
            # Single dataset scenario
            model_path, data_path = download_result
            return handle_request(model_service.load_model_and_datasets, 
                                model_path, data_path=data_path, target_column=target_column)
        else:
            # Separate datasets scenario
            model_path, train_path, test_path = download_result
            return handle_request(model_service.load_model_and_datasets, 
                                model_path, train_data_path=train_path, test_data_path=test_path, target_column=target_column)
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error loading from S3: {str(e)}")

@app.get("/analysis/overview", tags=["Analysis"])
async def get_overview(token: str = Depends(verify_token)):
    return handle_request(model_service.get_model_overview)

@app.get("/analysis/classification-stats", tags=["Analysis"])
async def get_classification_statistics(token: str = Depends(verify_token)):
    return handle_request(model_service.get_classification_stats)

@app.get("/analysis/feature-importance", tags=["Analysis"])
async def get_feature_importance(method: str = 'shap', token: str = Depends(verify_token)):
    return handle_request(model_service.get_feature_importance, method)

@app.get("/analysis/explain-instance/{instance_idx}", tags=["Analysis"])
async def explain_instance(instance_idx: int, token: str = Depends(verify_token)):
    return handle_request(model_service.explain_instance, instance_idx)

@app.post("/analysis/what-if", tags=["Analysis"])
async def perform_what_if(payload: Dict = Body(...), token: str = Depends(verify_token)):
    return handle_request(model_service.perform_what_if, payload.get("features"))

@app.get("/analysis/feature-dependence/{feature_name}", tags=["Analysis"])
async def get_feature_dependence(feature_name: str, token: str = Depends(verify_token)):
    return handle_request(model_service.get_feature_dependence, feature_name)

@app.get("/analysis/instances", tags=["Analysis"])
async def list_instances(sort_by: str = 'prediction', limit: int = 100, token: str = Depends(verify_token)):
    return handle_request(model_service.list_instances, sort_by, limit)

@app.get("/analysis/dataset-comparison", tags=["Analysis"])
async def get_dataset_comparison(token: str = Depends(verify_token)):
    return handle_request(model_service.get_dataset_comparison)

# --- New enterprise feature endpoints ---
@app.get("/api/features", tags=["Features"])
async def get_features_metadata(token: str = Depends(verify_token)):
    return handle_request(model_service.get_feature_metadata)

@app.post("/api/correlation", tags=["Features"])
async def post_correlation(payload: Dict = Body(...), token: str = Depends(verify_token)):
    selected: List[str] = payload.get("features") or []
    return handle_request(model_service.compute_correlation, selected)

@app.post("/api/feature-importance", tags=["Features"])
async def post_feature_importance(payload: Dict = Body(...), token: str = Depends(verify_token)):
    method = payload.get("method", "shap")
    sort_by = payload.get("sort_by", "importance")
    top_n = int(payload.get("top_n", 20))
    visualization = payload.get("visualization", "bar")
    return handle_request(model_service.compute_feature_importance_advanced, method, sort_by, top_n, visualization)

@app.get("/analysis/feature-interactions", tags=["Analysis"])
async def get_feature_interactions(feature1: str, feature2: str, token: str = Depends(verify_token)):
    return handle_request(model_service.get_feature_interactions, feature1, feature2)

@app.get("/analysis/decision-tree", tags=["Analysis"])
async def get_decision_tree(token: str = Depends(verify_token)):
    return handle_request(model_service.get_decision_tree)

# --- Section 2 APIs ---
@app.post("/api/roc-analysis", tags=["Classification"])
async def post_roc_analysis(token: str = Depends(verify_token)):
    return handle_request(model_service.roc_analysis)

@app.get("/api/roc-analysis", tags=["Classification"])
async def get_roc_analysis(token: str = Depends(verify_token)):
    return handle_request(model_service.roc_analysis)

@app.post("/api/threshold-analysis", tags=["Classification"])
async def post_threshold_analysis(num_thresholds: int = 50, token: str = Depends(verify_token)):
    return handle_request(model_service.threshold_analysis, num_thresholds)

# --- Section 3 API ---  
@app.post("/api/individual-prediction", tags=["Prediction"])
async def post_individual_prediction(payload: Dict = Body(...), token: str = Depends(verify_token)):
    instance_idx = int(payload.get("instance_idx", 0))
    return handle_request(model_service.individual_prediction, instance_idx)

# --- Section 4 APIs ---
@app.post("/api/partial-dependence", tags=["Dependence"])
async def post_partial_dependence(payload: Dict = Body(...), token: str = Depends(verify_token)):
    feature = payload.get("feature")
    if not feature:
        raise HTTPException(status_code=400, detail="Missing 'feature'")
    num_points = int(payload.get("num_points", 20))
    return handle_request(model_service.partial_dependence, feature, num_points)

@app.post("/api/shap-dependence", tags=["Dependence"])
async def post_shap_dependence(payload: Dict = Body(...), token: str = Depends(verify_token)):
    feature = payload.get("feature")
    if not feature:
        raise HTTPException(status_code=400, detail="Missing 'feature'")
    color_by = payload.get("color_by")
    return handle_request(model_service.shap_dependence, feature, color_by)

@app.post("/api/ice-plot", tags=["Dependence"])
async def post_ice_plot(payload: Dict = Body(...), token: str = Depends(verify_token)):
    feature = payload.get("feature")
    if not feature:
        raise HTTPException(status_code=400, detail="Missing 'feature'")
    num_points = int(payload.get("num_points", 20))
    num_instances = int(payload.get("num_instances", 20))
    return handle_request(model_service.ice_plot, feature, num_points, num_instances)

# --- Section 5 APIs ---
@app.post("/api/interaction-network", tags=["Interactions"])
async def post_interaction_network(payload: Dict = Body({}), token: str = Depends(verify_token)):
    top_k = int(payload.get("top_k", 30))
    sample_rows = int(payload.get("sample_rows", 200))
    return handle_request(model_service.interaction_network, top_k, sample_rows)

@app.post("/api/pairwise-analysis", tags=["Interactions"])
async def post_pairwise_analysis(payload: Dict = Body(...), token: str = Depends(verify_token)):
    f1 = payload.get("feature1")
    f2 = payload.get("feature2")
    if not f1 or not f2:
        raise HTTPException(status_code=400, detail="Missing 'feature1' or 'feature2'")
    color_by = payload.get("color_by")
    sample_size = int(payload.get("sample_size", 1000))
    return handle_request(model_service.pairwise_analysis, f1, f2, color_by, sample_size)

# --- AI Explanation Endpoint ---
@app.post("/analysis/explain-with-ai", tags=["AI Analysis"])
async def explain_with_ai(
    payload: Dict = Body(...),
    token: str = Depends(verify_token)
):
    """
    Generate an AI-powered explanation of the current analysis results.
    
    Expected payload:
    {
        "analysis_type": "overview|feature_importance|classification_stats|...",
        "analysis_data": {...}  # The data to be explained
    }
    """
    try:
        analysis_type = payload.get("analysis_type")
        analysis_data = payload.get("analysis_data", {})
        
        if not analysis_type:
            raise HTTPException(status_code=400, detail="Missing 'analysis_type' in payload")
        
        # Generate AI explanation
        explanation = ai_explanation_service.generate_explanation(analysis_data, analysis_type)
        
        return JSONResponse(status_code=200, content=explanation)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating AI explanation: {str(e)}")
