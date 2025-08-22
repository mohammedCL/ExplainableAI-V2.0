import os

class Settings:
    PROJECT_NAME: str = "ML Model Analysis Dashboard API"
    PROJECT_VERSION: str = "1.0.0"
    
    # Local storage for temporary files
    STORAGE_DIR: str = os.path.join(os.getcwd(), "storage")
    
    # S3 Configuration for data and model storage
    S3_ACCESS_TOKEN: str = os.getenv("S3_ACCESS_TOKEN")
    S3_ENDPOINT_URL: str = os.getenv("S3_ENDPOINT_URL", "http://xailoadbalancer-579761463.ap-south-1.elb.amazonaws.com/api/files_download/Classification")
    
    # AWS Bedrock Configuration
    AWS_REGION: str = os.getenv("REGION_LLM", "us-east-1")
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID_LLM", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY_LLM", "")
    AWS_SESSION_TOKEN: str = os.getenv("AWS_SESSION_TOKEN_LLM", "")
    
    # Ensure the storage directory exists
    os.makedirs(STORAGE_DIR, exist_ok=True)

settings = Settings()
