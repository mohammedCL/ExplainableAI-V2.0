import os

class Settings:
    PROJECT_NAME: str = "ML Model Analysis Dashboard API"
    PROJECT_VERSION: str = "1.0.0"
    
    # We will store uploaded files in a local directory for now
    STORAGE_DIR: str = os.path.join(os.getcwd(), "storage")
    
    # Ensure the storage directory exists
    os.makedirs(STORAGE_DIR, exist_ok=True)

settings = Settings()
