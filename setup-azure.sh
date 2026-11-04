#!/bin/bash

# Azure Setup Script for Expense Categorizer
# This script sets up the Azure resources needed for the expense categorizer

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
RESOURCE_GROUP="expense-categorizer-rg"
LOCATION="eastus"
APP_NAME="expense-categorizer-prod"
APP_PLAN="expense-categorizer-plan"
STORAGE_ACCOUNT="expensecategorizerstorage"
KEY_VAULT="expense-categorizer-kv"

echo -e "${GREEN}🚀 Setting up Azure resources for Expense Categorizer${NC}"

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo -e "${RED}❌ Azure CLI is not installed. Please install it first.${NC}"
    echo "Visit: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

# Check if logged in
if ! az account show &> /dev/null; then
    echo -e "${YELLOW}⚠️ Not logged in to Azure. Please run 'az login' first.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Azure CLI is ready${NC}"

# Create resource group
echo -e "${YELLOW}📦 Creating resource group...${NC}"
az group create \
    --name $RESOURCE_GROUP \
    --location $LOCATION \
    --output table

# Create storage account
echo -e "${YELLOW}💾 Creating storage account...${NC}"
az storage account create \
    --name $STORAGE_ACCOUNT \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --sku Standard_LRS \
    --output table

# Create App Service plan (Free tier)
echo -e "${YELLOW}🏗️ Creating App Service plan...${NC}"
az appservice plan create \
    --name $APP_PLAN \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --sku FREE \
    --is-linux \
    --output table

# Create web app
echo -e "${YELLOW}🌐 Creating web app...${NC}"
az webapp create \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --plan $APP_PLAN \
    --runtime "PYTHON|3.11" \
    --output table

# Configure app settings
echo -e "${YELLOW}⚙️ Configuring app settings...${NC}"

# Get storage connection string
STORAGE_CONNECTION_STRING=$(az storage account show-connection-string \
    --name $STORAGE_ACCOUNT \
    --resource-group $RESOURCE_GROUP \
    --query connectionString \
    --output tsv)

# Set app settings
az webapp config appsettings set \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --settings \
        AZURE_STORAGE_CONNECTION_STRING="$STORAGE_CONNECTION_STRING" \
        MLFLOW_TRACKING_URI="sqlite:///mlruns/mlflow.db" \
        PYTHONPATH="/home/site/wwwroot" \
        WEBSITES_ENABLE_APP_SERVICE_STORAGE="true" \
        --output table

# Configure deployment source (GitHub)
echo -e "${YELLOW}🔗 Configuring deployment source...${NC}"
echo "Please configure your deployment source manually:"
echo "1. Go to Azure Portal > App Services > $APP_NAME"
echo "2. Go to Deployment Center"
echo "3. Choose GitHub as source"
echo "4. Connect your repository"
echo "5. Set branch to 'main'"
echo "6. Set startup command to: 'gunicorn --bind=0.0.0.0 --timeout 600 src.app:app'"

# Create Key Vault for secrets
echo -e "${YELLOW}🔐 Creating Key Vault...${NC}"
az keyvault create \
    --name $KEY_VAULT \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --output table

# Get web app managed identity
echo -e "${YELLOW}🔑 Configuring managed identity...${NC}"
az webapp identity assign \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --output table

# Get the principal ID
PRINCIPAL_ID=$(az webapp identity show \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --query principalId \
    --output tsv)

# Grant Key Vault access to the web app
az keyvault set-policy \
    --name $KEY_VAULT \
    --object-id $PRINCIPAL_ID \
    --secret-permissions get list \
    --output table

echo -e "${GREEN}✅ Azure setup completed!${NC}"
echo ""
echo -e "${YELLOW}📋 Next steps:${NC}"
echo "1. Add your secrets to Key Vault:"
echo "   - AZURE_CLIENT_ID"
echo "   - AZURE_CLIENT_SECRET" 
echo "   - AZURE_TENANT_ID"
echo "   - OPENAI_API_KEY"
echo ""
echo "2. Update your app settings to use Key Vault references:"
echo "   - Go to App Service > Configuration"
echo "   - Add new settings with Key Vault references"
echo ""
echo "3. Configure GitHub Actions secrets:"
echo "   - AZURE_CREDENTIALS (service principal JSON)"
echo "   - AZURE_WEBAPP_PUBLISH_PROFILE"
echo ""
echo -e "${GREEN}🎉 Your expense categorizer is ready for deployment!${NC}"

