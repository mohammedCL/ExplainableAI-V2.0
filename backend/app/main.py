from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Depends, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import Dict

from app.core.config import settings
from app.core.auth import verify_token
from app.services.model_service import ModelService

app = FastAPI(title=settings.PROJECT_NAME, version=settings.PROJECT_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_service = ModelService()

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

@app.get("/analysis/feature-interactions", tags=["Analysis"])
async def get_feature_interactions(feature1: str, feature2: str, token: str = Depends(verify_token)):
    return handle_request(model_service.get_feature_interactions, feature1, feature2)

@app.get("/analysis/decision-tree", tags=["Analysis"])
async def get_decision_tree(token: str = Depends(verify_token)):
    return handle_request(model_service.get_decision_tree)