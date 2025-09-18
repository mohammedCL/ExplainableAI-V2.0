from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List
import numpy as np
import math

from app.core.config import settings
from app.core.auth import verify_token
from app.services.model_service import ModelService
from app.services.ai_explanation_service import AIExplanationService

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


# Auto-load functionality disabled as requested by user
# User will upload model and dataset through the frontend interface

def sanitize_for_json(obj):
    """Recursively sanitize an object to ensure JSON serialization compatibility."""
    if obj is None:
        return None
    elif isinstance(obj, (bool, str)):
        return obj
    elif isinstance(obj, (int, float)):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    else:
        # For any other type, try to convert to string
        return str(obj)

# --- Utility Function for Error Handling ---
def handle_request(service_func, *args, **kwargs):
    import json
    import traceback
    
    try:
        result = service_func(*args, **kwargs)
        
        # Sanitize the result before JSON serialization
        sanitized_result = sanitize_for_json(result)
        
        # Test JSON serialization
        try:
            json_str = json.dumps(sanitized_result)
            print(f"✅ Service result successfully serialized to JSON ({len(json_str)} characters)")
        except (TypeError, ValueError) as json_error:
            print(f"❌ JSON serialization error even after sanitization: {json_error}")
            print(f"🔍 Result type: {type(sanitized_result)}")
            
            # Find problematic parts in the sanitized result
            if isinstance(sanitized_result, dict):
                for key, value in sanitized_result.items():
                    try:
                        json.dumps({key: value})
                    except Exception as e:
                        print(f"❌ Problematic key '{key}': {value} (type: {type(value)}) - {e}")
            
            raise ValueError(f"Response contains values that cannot be serialized to JSON: {json_error}")
        
        return JSONResponse(status_code=200, content=sanitized_result)
        
    except ValueError as e:
        print(f"❌ ValueError in handle_request: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"💥 Unexpected error in handle_request: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

# --- API Endpoints ---

@app.get("/", tags=["Root"])
def read_root():
    return {"message": f"Welcome to the {settings.PROJECT_NAME}"}

from fastapi import APIRouter, Depends, HTTPException, requests
from dotenv import load_dotenv
import requests
import os



@app.get("/api/files")
def get_s3_file_metadata():
    """
    Lists files and models from the external S3 API and returns their metadata (name, URL, folder).
    Separates files and models based on the folder field.
    """
    file_api = "http://xailoadbalancer-579761463.ap-south-1.elb.amazonaws.com/api/files_download"
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0YXFpdWRkaW4ubW9oYW1tZWRAY2lycnVzbGFicy5pbyIsInVzZXJfaWQiOjQyLCJyb2xlcyI6W10sInBlcm1pc3Npb25zIjpbXSwiZXhwIjoxNzU3MDc5NDA0fQ.xOuA3zFw7-qjiTTI5Fl0cU0-2YWRbAkgKTFQCpJH76Y"
    EXTERNAL_S3_API_URL = f"{file_api}/Classification"
    headers = {
        "Authorization": f"Bearer {token}"
    }
    try:
        response = requests.get(EXTERNAL_S3_API_URL, headers=headers)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        json_data = response.json()
        all_items = json_data.get("files", [])
        
        # Separate files and models based on folder
        files = [item for item in all_items if item.get("folder") == "files"]
        models = [item for item in all_items if item.get("folder") == "models"]
        
        return {
            "files": files,
            "models": models
        }
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to external S3 API: {e}")
        return None
    except Exception as e:
        print(f"Error processing external S3 API response: {e}")
        return None

from typing import Dict
from fastapi import APIRouter, Depends, Body, HTTPException
from pydantic import BaseModel

class LoadDataRequest(BaseModel):
    model: str
    train_dataset: str = None
    test_dataset: str = None
    target_column: str = "target"


@app.post("/load")
async def load_data(payload: LoadDataRequest):
    
    try:
        model_name = payload.model
        train_dataset = payload.train_dataset
        test_dataset = payload.test_dataset
        target_column = payload.target_column

        if not model_name:
            raise HTTPException(status_code=400, detail="Missing model name")

        return handle_request(model_service.load_model_and_datasets, model_path=model_name, train_data_path=train_dataset, test_data_path=test_dataset, target_column=target_column)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

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

# @app.get("/analysis/dataset-comparison", tags=["Analysis"])
# async def get_dataset_comparison(token: str = Depends(verify_token)):
#     return handle_request(model_service.get_dataset_comparison)

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
