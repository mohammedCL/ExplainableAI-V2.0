from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Depends, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import Dict
from typing import List

from core.config import settings
from core.auth import verify_token
from services.model_service import ModelService

app = FastAPI(title=settings.PROJECT_NAME, version=settings.PROJECT_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_service = ModelService()

# # Auto-load model and data on startup
# @app.on_event("startup")
# async def startup_event():
#     try:
#         model_path = os.path.join(settings.STORAGE_DIR, "cancer_model.joblib")
#         data_path = os.path.join(settings.STORAGE_DIR, "breast_cancer_dataset.csv")
#         target_column = "target"
        
#         if os.path.exists(model_path) and os.path.exists(data_path):
#             print(f"Auto-loading model from {model_path}")
#             print(f"Auto-loading data from {data_path}")
#             result = model_service.load_model_and_data(model_path, data_path, target_column)
#             print("✅ Model and data loaded successfully on startup!")
#         else:
#             print("❌ Model or data files not found, waiting for upload...")
#     except Exception as e:
#         print(f"❌ Error auto-loading model: {e}")
#         import traceback
#         traceback.print_exc()

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

@app.post("/upload/model-and-data", tags=["Setup"])
async def upload_model_and_data(
    token: str = Depends(verify_token),
    model_file: UploadFile = File(..., description="A .joblib model file."),
    data_file: UploadFile = File(..., description="A .csv dataset file."),
    target_column: str = Form(..., description="The name of the target variable column in the CSV.")
):
    model_path = os.path.join(settings.STORAGE_DIR, model_file.filename)
    with open(model_path, "wb") as buffer:
        buffer.write(await model_file.read())
    
    data_path = os.path.join(settings.STORAGE_DIR, data_file.filename)
    with open(data_path, "wb") as buffer:
        buffer.write(await data_file.read())

    return handle_request(model_service.load_model_and_data, model_path, data_path, target_column)

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