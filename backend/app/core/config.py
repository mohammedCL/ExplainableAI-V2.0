import os

class Settings:
    PROJECT_NAME: str = "ML Model Analysis Dashboard API"
    PROJECT_VERSION: str = "1.0.0"
    
    # We will store uploaded files in a local directory for now
    STORAGE_DIR: str = os.path.join(os.getcwd(), "storage")
    
    # AWS Bedrock Configuration
    AWS_REGION: str = os.getenv("REGION_LLM", "us-east-1")
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID_LLM", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY_LLM", "")
    AWS_SESSION_TOKEN: str = os.getenv("AWS_SESSION_TOKEN_LLM", "")
    
    # Ensure the storage directory exists
    os.makedirs(STORAGE_DIR, exist_ok=True)

settings = Settings()
