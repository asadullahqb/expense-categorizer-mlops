"""
Azure Setup Script for Expense Categorizer

This script helps set up the expense categorizer on Azure using free services.
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

class AzureSetup:
    """Handles Azure setup for the expense categorizer"""
    
    def __init__(self):
        self.resource_group = "expense-categorizer-rg"
        self.app_service_plan = "expense-categorizer-plan"
        self.web_app_name = "expense-categorizer-app"
        self.storage_account = "expensecategorizerstorage"
        self.location = "East US"  # Free tier available
        
    def check_prerequisites(self) -> bool:
        """Check if Azure CLI is installed and user is logged in"""
        try:
            # Check Azure CLI
            result = subprocess.run(["az", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                print("❌ Azure CLI not found. Please install Azure CLI first.")
                print("   Download from: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli")
                return False
            
            # Check if logged in
            result = subprocess.run(["az", "account", "show"], capture_output=True, text=True)
            if result.returncode != 0:
                print("❌ Not logged in to Azure. Please run 'az login' first.")
                return False
            
            print("✅ Azure CLI is installed and user is logged in")
            return True
            
        except FileNotFoundError:
            print("❌ Azure CLI not found. Please install Azure CLI first.")
            return False
    
    def create_resource_group(self) -> bool:
        """Create Azure resource group"""
        try:
            print(f"📦 Creating resource group: {self.resource_group}")
            
            cmd = [
                "az", "group", "create",
                "--name", self.resource_group,
                "--location", self.location
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Failed to create resource group: {result.stderr}")
                return False
            
            print("✅ Resource group created successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error creating resource group: {str(e)}")
            return False
    
    def create_app_service_plan(self) -> bool:
        """Create Azure App Service plan (free tier)"""
        try:
            print(f"📋 Creating App Service plan: {self.app_service_plan}")
            
            cmd = [
                "az", "appservice", "plan", "create",
                "--name", self.app_service_plan,
                "--resource-group", self.resource_group,
                "--sku", "F1",  # Free tier
                "--is-linux"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Failed to create App Service plan: {result.stderr}")
                return False
            
            print("✅ App Service plan created successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error creating App Service plan: {str(e)}")
            return False
    
    def create_web_app(self) -> bool:
        """Create Azure Web App"""
        try:
            print(f"🌐 Creating Web App: {self.web_app_name}")
            
            cmd = [
                "az", "webapp", "create",
                "--name", self.web_app_name,
                "--resource-group", self.resource_group,
                "--plan", self.app_service_plan,
                "--runtime", "PYTHON|3.11"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Failed to create Web App: {result.stderr}")
                return False
            
            print("✅ Web App created successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error creating Web App: {str(e)}")
            return False
    
    def create_storage_account(self) -> bool:
        """Create Azure Storage Account (free tier)"""
        try:
            print(f"💾 Creating Storage Account: {self.storage_account}")
            
            cmd = [
                "az", "storage", "account", "create",
                "--name", self.storage_account,
                "--resource-group", self.resource_group,
                "--location", self.location,
                "--sku", "Standard_LRS"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Failed to create Storage Account: {result.stderr}")
                return False
            
            print("✅ Storage Account created successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error creating Storage Account: {str(e)}")
            return False
    
    def configure_app_settings(self) -> bool:
        """Configure Web App settings"""
        try:
            print("⚙️ Configuring Web App settings...")
            
            # Get storage account key
            cmd = [
                "az", "storage", "account", "keys", "list",
                "--account-name", self.storage_account,
                "--resource-group", self.resource_group
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Failed to get storage account key: {result.stderr}")
                return False
            
            storage_key = json.loads(result.stdout)[0]['value']
            
            # Configure app settings
            settings = [
                f"AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName={self.storage_account};AccountKey={storage_key};EndpointSuffix=core.windows.net",
                "MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db",
                "PYTHONPATH=/home/site/wwwroot",
                "WEBSITES_ENABLE_APP_SERVICE_STORAGE=true"
            ]
            
            for setting in settings:
                cmd = [
                    "az", "webapp", "config", "appsettings", "set",
                    "--name", self.web_app_name,
                    "--resource-group", self.resource_group,
                    "--settings", setting
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"❌ Failed to set app setting: {result.stderr}")
                    return False
            
            print("✅ Web App settings configured successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error configuring app settings: {str(e)}")
            return False
    
    def create_deployment_script(self) -> bool:
        """Create deployment script for GitHub Actions"""
        try:
            print("📝 Creating deployment script...")
            
            script_content = f"""#!/bin/bash
# Azure deployment script for Expense Categorizer

# Build and push Docker image
docker build -t {self.web_app_name} .
docker tag {self.web_app_name} {self.web_app_name}.azurecr.io/{self.web_app_name}:latest

# Deploy to Azure Web App
az webapp config container set \\
    --name {self.web_app_name} \\
    --resource-group {self.resource_group} \\
    --docker-custom-image-name {self.web_app_name}.azurecr.io/{self.web_app_name}:latest

echo "Deployment complete!"
"""
            
            script_path = Path("scripts/deploy.sh")
            script_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Make script executable
            os.chmod(script_path, 0o755)
            
            print("✅ Deployment script created")
            return True
            
        except Exception as e:
            print(f"❌ Error creating deployment script: {str(e)}")
            return False
    
    def setup_github_secrets(self) -> Dict[str, str]:
        """Generate GitHub secrets configuration"""
        print("🔐 GitHub Secrets Configuration:")
        print("Add these secrets to your GitHub repository:")
        print("")
        
        secrets = {
            "AZURE_CREDENTIALS": "Service principal credentials (JSON)",
            "AZURE_WEBAPP_PUBLISH_PROFILE": "Web App publish profile",
            "AZURE_CLIENT_ID": "Azure AD application client ID",
            "AZURE_CLIENT_SECRET": "Azure AD application client secret",
            "AZURE_TENANT_ID": "Azure AD tenant ID",
            "OPENAI_API_KEY": "OpenAI API key for LLM categorization"
        }
        
        for secret, description in secrets.items():
            print(f"  {secret}: {description}")
        
        print("")
        print("To get Azure credentials:")
        print("1. Create service principal: az ad sp create-for-rbac --name expense-categorizer-sp")
        print("2. Get publish profile: az webapp deployment list-publishing-profiles --name expense-categorizer-app --resource-group expense-categorizer-rg")
        
        return secrets
    
    def run_full_setup(self) -> bool:
        """Run the complete Azure setup"""
        print("🚀 Starting Azure setup for Expense Categorizer")
        print("=" * 50)
        
        # Check prerequisites
        if not self.check_prerequisites():
            return False
        
        # Create resources
        steps = [
            ("Creating resource group", self.create_resource_group),
            ("Creating App Service plan", self.create_app_service_plan),
            ("Creating Web App", self.create_web_app),
            ("Creating Storage Account", self.create_storage_account),
            ("Configuring app settings", self.configure_app_settings),
            ("Creating deployment script", self.create_deployment_script)
        ]
        
        for step_name, step_func in steps:
            print(f"\n{step_name}...")
            if not step_func():
                print(f"❌ Setup failed at: {step_name}")
                return False
        
        # Show GitHub secrets
        print("\n" + "=" * 50)
        self.setup_github_secrets()
        
        print("\n✅ Azure setup complete!")
        print(f"🌐 Web App URL: https://{self.web_app_name}.azurewebsites.net")
        print(f"📊 Resource Group: {self.resource_group}")
        
        return True

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Azure Setup for Expense Categorizer")
    parser.add_argument("--full", action="store_true", help="Run full setup")
    parser.add_argument("--check", action="store_true", help="Check prerequisites only")
    parser.add_argument("--secrets", action="store_true", help="Show GitHub secrets configuration")
    
    args = parser.parse_args()
    
    setup = AzureSetup()
    
    if args.check:
        setup.check_prerequisites()
    elif args.secrets:
        setup.setup_github_secrets()
    elif args.full:
        setup.run_full_setup()
    else:
        print("Please specify an option: --full, --check, or --secrets")

if __name__ == "__main__":
    main()

