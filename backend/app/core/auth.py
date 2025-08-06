from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Reusable security dependency
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Placeholder token verification.
    
    In a real application, you would decode the JWT token, check its
    validity, expiration, and scopes.

    For now, we will accept any token that is not empty.
    """
    if not credentials.credentials:
        raise HTTPException(status_code=401, detail="Invalid or missing access token")
    
    # In a real app:
    # is_valid = your_real_token_validation_logic(credentials.credentials)
    # if not is_valid:
    #     raise HTTPException(status_code=401, detail="Invalid or expired access token")

    print(f"Auth placeholder: Successfully 'validated' token.")
    return credentials.credentials
