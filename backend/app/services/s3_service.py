import os
import requests
from typing import Optional, Tuple, Dict
from app.core.config import settings


class S3Service:
    """
    Service to handle S3 file downloads using access token and endpoint URL.
    """
    
    def __init__(self):
        self.access_token = settings.S3_ACCESS_TOKEN
        self.endpoint_url = settings.S3_ENDPOINT_URL
        self.listing_url = "http://xailoadbalancer-579761463.ap-south-1.elb.amazonaws.com/api/files_download/Classification"
        
    def _get_headers(self) -> dict:
        """Get headers with authorization token."""
        return {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
    
    def _get_file_list(self) -> Dict[str, str]:
        """Get the list of available files and their download URLs."""
        try:
            headers = self._get_headers()
            response = requests.get(self.listing_url, headers=headers)
            response.raise_for_status()
            
            file_data = response.json()
            file_map = {}
            
            if 'files' in file_data:
                for file_info in file_data['files']:
                    file_name = file_info.get('file_name')
                    folder = file_info.get('folder', '')
                    download_url = file_info.get('url')
                    
                    if file_name and download_url:
                        # Create both with and without folder prefix for flexibility
                        file_map[file_name] = download_url
                        if folder:
                            file_map[f"{folder}/{file_name}"] = download_url
                            
            logging.info(f"📋 Found {len(file_map)} files in S3 bucket")
            return file_map
            
        except Exception as e:
            logging.error(f"❌ Failed to get file list: {str(e)}")
            return {}

    def download_file(self, s3_key: str, local_path: str) -> bool:
        """
        Download a file from S3 to local storage using the file listing.
        
        Args:
            s3_key: The S3 object key/path
            local_path: Local path where the file should be saved
            
        Returns:
            bool: True if download successful, False otherwise
        """
        try:
            if not self.access_token:
                raise ValueError("S3 access token must be configured")
            
            # Get file list to find the actual download URL
            file_map = self._get_file_list()
            
            # Try to find the file in the map
            download_url = None
            
            # Try exact match first
            if s3_key in file_map:
                download_url = file_map[s3_key]
            else:
                # Try without folder prefix
                file_name = os.path.basename(s3_key)
                if file_name in file_map:
                    download_url = file_map[file_name]
                else:
                    # Try different folder combinations
                    for key, url in file_map.items():
                        if key.endswith(file_name) or file_name in key:
                            download_url = url
                            break
            
            if not download_url:
                available_files = list(file_map.keys())
                print(f"❌ File '{s3_key}' not found. Available files: {available_files}")
                return False
            
            print(f"🔗 Using download URL for {s3_key}: {download_url[:100]}...")
            
            # Download the file using the pre-signed URL
            # Note: Don't add authorization headers for pre-signed URLs
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Save file to local storage
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Verify file was downloaded correctly
            file_size = os.path.getsize(local_path)
            if file_size < 100:  # Suspiciously small, might be an error page
                with open(local_path, 'r') as f:
                    content = f.read(200)
                    if 'html' in content.lower() or 'doctype' in content.lower():
                        print(f"❌ Downloaded HTML instead of file for {s3_key}")
                        return False
            
            print(f"✅ Successfully downloaded {s3_key} to {local_path} ({file_size:,} bytes)")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to download {s3_key}: HTTP error - {str(e)}")
            return False
        except Exception as e:
            print(f"❌ Failed to download {s3_key}: {str(e)}")
            return False
    
    def download_model_and_datasets(self, model_s3_key: str, data_s3_key: str = None, train_s3_key: str = None, test_s3_key: str = None) -> Tuple[Optional[str], ...]:
        """
        Download model and dataset(s) from S3. Supports both single dataset and separate train/test scenarios.
        
        Args:
            model_s3_key: S3 key for the model file
            data_s3_key: S3 key for single dataset file (optional)
            train_s3_key: S3 key for training dataset file (optional)
            test_s3_key: S3 key for test dataset file (optional)
            
        Returns:
            For single dataset: Tuple(model_path, data_path) or (None, None)
            For separate datasets: Tuple(model_path, train_path, test_path) or (None, None, None)
        """
        try:
            # Validate input parameters
            if data_s3_key and (train_s3_key or test_s3_key):
                raise ValueError("Provide either data_s3_key OR train_s3_key+test_s3_key, not both")
            
            if not data_s3_key and not (train_s3_key and test_s3_key):
                raise ValueError("Must provide either data_s3_key OR both train_s3_key and test_s3_key")
            
            # Determine scenario and prepare download list
            downloads = []
            
            # Model file (always required)
            model_filename = os.path.basename(model_s3_key)
            model_local_path = os.path.join(settings.STORAGE_DIR, model_filename)
            downloads.append((model_s3_key, model_local_path))
            
            if data_s3_key:
                # Single dataset scenario
                data_filename = os.path.basename(data_s3_key)
                data_local_path = os.path.join(settings.STORAGE_DIR, data_filename)
                downloads.append((data_s3_key, data_local_path))
                return_paths = [model_local_path, data_local_path]
                none_return = (None, None)
            else:
                # Separate train/test datasets scenario
                train_filename = f"train_{os.path.basename(train_s3_key)}"
                test_filename = f"test_{os.path.basename(test_s3_key)}"
                
                train_local_path = os.path.join(settings.STORAGE_DIR, train_filename)
                test_local_path = os.path.join(settings.STORAGE_DIR, test_filename)
                
                downloads.append((train_s3_key, train_local_path))
                downloads.append((test_s3_key, test_local_path))
                return_paths = [model_local_path, train_local_path, test_local_path]
                none_return = (None, None, None)
            
            # Download all files
            downloaded_files = []
            for s3_key, local_path in downloads:
                if self.download_file(s3_key, local_path):
                    downloaded_files.append(local_path)
                else:
                    # Clean up any successfully downloaded files if one fails
                    for file_path in downloaded_files:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    return none_return
            
            return tuple(return_paths)
            
        except Exception as e:
            print(f"❌ Failed to download model and datasets: {str(e)}")
            if data_s3_key:
                return None, None
            else:
                return None, None, None
    
    # Backward compatibility wrapper methods
    def download_model_and_data(self, model_s3_key: str, data_s3_key: str) -> Tuple[Optional[str], Optional[str]]:
        """Legacy method for single dataset download."""
        return self.download_model_and_datasets(model_s3_key, data_s3_key=data_s3_key)
    
    def download_model_and_separate_datasets(self, model_s3_key: str, train_s3_key: str, test_s3_key: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Legacy method for separate datasets download."""
        return self.download_model_and_datasets(model_s3_key, train_s3_key=train_s3_key, test_s3_key=test_s3_key)