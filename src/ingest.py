"""
OneDrive Data Ingestion Module

This module handles authentication and data retrieval from OneDrive using Microsoft Graph API.
Optimized for free Microsoft Azure services.
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import msal
import requests
from pathlib import Path
import mlflow
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OneDriveIngestion:
    """Handles OneDrive data ingestion using Microsoft Graph API"""
    
    def __init__(self):
        self.client_id = os.getenv('AZURE_CLIENT_ID')
        self.client_secret = os.getenv('AZURE_CLIENT_SECRET')
        self.tenant_id = os.getenv('AZURE_TENANT_ID')
        self.authority = f"https://login.microsoftonline.com/{self.tenant_id}"
        self.scope = ["https://graph.microsoft.com/.default"]
        self.access_token = None
        
    def authenticate(self) -> bool:
        """Authenticate with Microsoft Graph API using client credentials flow"""
        try:
            app = msal.ConfidentialClientApplication(
                self.client_id,
                authority=self.authority,
                client_credential=self.client_secret
            )
            
            result = app.acquire_token_silent(self.scope, account=None)
            if not result:
                result = app.acquire_token_for_client(scopes=self.scope)
            
            if "access_token" in result:
                self.access_token = result["access_token"]
                print("✅ Successfully authenticated with Microsoft Graph API")
                return True
            else:
                print(f"❌ Authentication failed: {result.get('error_description', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"❌ Authentication error: {str(e)}")
            return False
    
    def get_headers(self) -> Dict[str, str]:
        """Get headers for API requests"""
        return {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
    
    def list_files(self, folder_path: str = "/") -> List[Dict]:
        """List files in OneDrive folder"""
        if not self.access_token:
            raise Exception("Not authenticated. Call authenticate() first.")
        
        try:
            # URL encode the folder path
            encoded_path = folder_path.replace("/", "%2F")
            url = f"https://graph.microsoft.com/v1.0/me/drive/root:{encoded_path}:/children"
            
            response = requests.get(url, headers=self.get_headers())
            response.raise_for_status()
            
            data = response.json()
            files = []
            
            for item in data.get('value', []):
                if 'file' in item:  # Only files, not folders
                    files.append({
                        'id': item['id'],
                        'name': item['name'],
                        'size': item.get('size', 0),
                        'modified': item.get('lastModifiedDateTime'),
                        'download_url': item.get('@microsoft.graph.downloadUrl')
                    })
            
            print(f"📁 Found {len(files)} files in {folder_path}")
            return files
            
        except Exception as e:
            print(f"❌ Error listing files: {str(e)}")
            return []
    
    def download_file(self, file_info: Dict, local_path: str) -> bool:
        """Download a file from OneDrive"""
        try:
            if file_info.get('download_url'):
                response = requests.get(file_info['download_url'])
                response.raise_for_status()
                
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"📥 Downloaded: {file_info['name']}")
                return True
            else:
                print(f"❌ No download URL for {file_info['name']}")
                return False
                
        except Exception as e:
            print(f"❌ Error downloading {file_info['name']}: {str(e)}")
            return False
    
    def ingest_transaction_files(self, target_folder: str = "/Finance/Transactions") -> Dict:
        """
        Main ingestion method - downloads transaction files from OneDrive
        
        Args:
            target_folder: OneDrive folder path containing transaction files
            
        Returns:
            Dict with ingestion metadata and file paths
        """
        if not self.authenticate():
            raise Exception("Failed to authenticate with Microsoft Graph API")
        
        # Create local directory for raw data
        raw_data_dir = Path("data/raw")
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Get timestamp for this ingestion run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # List files in target folder
        files = self.list_files(target_folder)
        
        if not files:
            print(f"⚠️ No files found in {target_folder}")
            return {
                'timestamp': timestamp,
                'files_processed': 0,
                'files_downloaded': 0,
                'local_paths': [],
                'errors': []
            }
        
        # Filter for common transaction file types
        transaction_extensions = ['.csv', '.xlsx', '.xls', '.json']
        transaction_files = [
            f for f in files 
            if any(f['name'].lower().endswith(ext) for ext in transaction_extensions)
        ]
        
        print(f"💰 Found {len(transaction_files)} transaction files")
        
        # Download files
        downloaded_files = []
        errors = []
        
        for file_info in transaction_files:
            local_filename = f"{timestamp}_{file_info['name']}"
            local_path = raw_data_dir / local_filename
            
            if self.download_file(file_info, str(local_path)):
                downloaded_files.append({
                    'original_name': file_info['name'],
                    'local_path': str(local_path),
                    'size': file_info['size'],
                    'modified': file_info['modified']
                })
            else:
                errors.append(f"Failed to download {file_info['name']}")
        
        # Log ingestion metadata to MLflow
        with mlflow.start_run(run_name=f"ingestion_{timestamp}"):
            mlflow.log_param("target_folder", target_folder)
            mlflow.log_param("files_found", len(files))
            mlflow.log_param("transaction_files", len(transaction_files))
            mlflow.log_param("files_downloaded", len(downloaded_files))
            mlflow.log_metric("total_size_bytes", sum(f['size'] for f in downloaded_files))
            
            if errors:
                mlflow.log_param("errors", errors)
        
        result = {
            'timestamp': timestamp,
            'files_processed': len(transaction_files),
            'files_downloaded': len(downloaded_files),
            'local_paths': [f['local_path'] for f in downloaded_files],
            'file_metadata': downloaded_files,
            'errors': errors
        }
        
        print(f"✅ Ingestion complete: {len(downloaded_files)} files downloaded")
        return result

def main():
    """Main function for testing the ingestion module"""
    ingester = OneDriveIngestion()
    
    try:
        result = ingester.ingest_transaction_files()
        print("\n📊 Ingestion Summary:")
        print(f"Files processed: {result['files_processed']}")
        print(f"Files downloaded: {result['files_downloaded']}")
        print(f"Errors: {len(result['errors'])}")
        
        if result['errors']:
            print("\n❌ Errors:")
            for error in result['errors']:
                print(f"  - {error}")
                
    except Exception as e:
        print(f"❌ Ingestion failed: {str(e)}")

if __name__ == "__main__":
    main()

