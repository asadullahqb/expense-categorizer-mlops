#!/bin/bash

# Deployment Script for Expense Categorizer
# This script handles local and cloud deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="expense-categorizer"
DOCKER_IMAGE="expense-categorizer:latest"
AZURE_WEBAPP_NAME="expense-categorizer-prod"

echo -e "${BLUE}🚀 Expense Categorizer Deployment Script${NC}"

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  local       Deploy locally with Docker"
    echo "  azure       Deploy to Azure App Service"
    echo "  test        Run tests and linting"
    echo "  pipeline    Run the complete Prefect pipeline"
    echo "  monitor     Run monitoring checks"
    echo "  help        Show this help message"
}

# Function to run tests
run_tests() {
    echo -e "${YELLOW}🧪 Running tests and linting...${NC}"
    
    # Install dependencies
    pip install -r requirements.txt
    pip install pytest pytest-cov flake8 black
    
    # Run linting
    echo -e "${YELLOW}📝 Running linting...${NC}"
    flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
    black --check src/
    
    # Run tests
    echo -e "${YELLOW}🔬 Running tests...${NC}"
    pytest tests/ -v --cov=src --cov-report=xml
    
    echo -e "${GREEN}✅ Tests completed successfully${NC}"
}

# Function to deploy locally
deploy_local() {
    echo -e "${YELLOW}🐳 Deploying locally with Docker...${NC}"
    
    # Build Docker image
    echo -e "${YELLOW}🔨 Building Docker image...${NC}"
    docker build -t $DOCKER_IMAGE .
    
    # Stop existing containers
    echo -e "${YELLOW}🛑 Stopping existing containers...${NC}"
    docker-compose down || true
    
    # Start services
    echo -e "${YELLOW}🚀 Starting services...${NC}"
    docker-compose up -d
    
    # Wait for services to be ready
    echo -e "${YELLOW}⏳ Waiting for services to start...${NC}"
    sleep 30
    
    # Test endpoints
    echo -e "${YELLOW}🔍 Testing endpoints...${NC}"
    curl -f http://localhost:8000/health || {
        echo -e "${RED}❌ Health check failed${NC}"
        exit 1
    }
    
    echo -e "${GREEN}✅ Local deployment successful${NC}"
    echo "🌐 API: http://localhost:8000"
    echo "📊 MLflow: http://localhost:5000"
}

# Function to deploy to Azure
deploy_azure() {
    echo -e "${YELLOW}☁️ Deploying to Azure...${NC}"
    
    # Check if Azure CLI is installed
    if ! command -v az &> /dev/null; then
        echo -e "${RED}❌ Azure CLI is not installed${NC}"
        exit 1
    fi
    
    # Check if logged in
    if ! az account show &> /dev/null; then
        echo -e "${RED}❌ Not logged in to Azure. Please run 'az login' first${NC}"
        exit 1
    fi
    
    # Build and push Docker image
    echo -e "${YELLOW}🔨 Building and pushing Docker image...${NC}"
    docker build -t $DOCKER_IMAGE .
    
    # Deploy to Azure App Service
    echo -e "${YELLOW}🚀 Deploying to Azure App Service...${NC}"
    az webapp config container set \
        --name $AZURE_WEBAPP_NAME \
        --resource-group expense-categorizer-rg \
        --docker-custom-image-name $DOCKER_IMAGE
    
    # Test deployment
    echo -e "${YELLOW}🔍 Testing deployment...${NC}"
    sleep 30
    curl -f https://$AZURE_WEBAPP_NAME.azurewebsites.net/health || {
        echo -e "${RED}❌ Deployment test failed${NC}"
        exit 1
    }
    
    echo -e "${GREEN}✅ Azure deployment successful${NC}"
    echo "🌐 App URL: https://$AZURE_WEBAPP_NAME.azurewebsites.net"
}

# Function to run pipeline
run_pipeline() {
    echo -e "${YELLOW}🔄 Running Prefect pipeline...${NC}"
    
    # Install Prefect
    pip install prefect
    
    # Run pipeline
    python src/pipeline.py
    
    echo -e "${GREEN}✅ Pipeline completed${NC}"
}

# Function to run monitoring
run_monitoring() {
    echo -e "${YELLOW}📊 Running monitoring checks...${NC}"
    
    python src/monitoring.py
    
    echo -e "${GREEN}✅ Monitoring completed${NC}"
}

# Main script logic
case "${1:-help}" in
    "local")
        run_tests
        deploy_local
        ;;
    "azure")
        run_tests
        deploy_azure
        ;;
    "test")
        run_tests
        ;;
    "pipeline")
        run_pipeline
        ;;
    "monitor")
        run_monitoring
        ;;
    "help"|*)
        show_usage
        ;;
esac

