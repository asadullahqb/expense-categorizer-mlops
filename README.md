# Expense Categorizer MLOps

An end-to-end MLOps pipeline that ingests transactions from OneDrive, categorizes them using ML/LLM, and deploys as a service with monitoring.

## Features

- 🔄 **Automated Data Ingestion**: Pulls transaction data from OneDrive using Microsoft Graph API
- 🤖 **Multi-Model Categorization**: Rule-based, LLM (GPT), and ML classification
- 📊 **Experiment Tracking**: MLflow integration for model versioning and metrics
- 🚀 **Cloud Deployment**: Azure App Service deployment with CI/CD
- 📈 **Monitoring**: Real-time monitoring and alerting
- 🔧 **Orchestration**: Prefect pipeline automation
- 🐳 **Containerization**: Docker support for easy deployment

## Project Structure

```
expense-categorizer-mlops/
├── data/
│   ├── raw/               # OneDrive ingested files
│   └── processed/         # cleaned & categorized
├── notebooks/             # exploration
├── src/
│   ├── ingest.py          # OneDrive ingestion
│   ├── preprocess.py      # cleaning, formatting
│   ├── categorize.py      # GPT/ML categorizer
│   ├── train_model.py     # ML classification
│   ├── app.py             # FastAPI service
│   ├── pipeline.py        # Prefect orchestration
│   └── monitoring.py      # monitoring & alerting
├── tests/                 # unit tests
├── mlruns/                # MLflow experiments
├── models/                # trained models
├── logs/                  # monitoring logs
├── .github/workflows/     # CI/CD pipelines
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Quick Start

### Local Development

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Environment Variables**
   ```bash
   cp env.example .env
   # Edit .env with your credentials
   ```

3. **Run the Application**
   ```bash
   uvicorn src.app:app --reload
   ```

### Docker Development

1. **Build and Run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Access the Application**
   - API: http://localhost:8000
   - MLflow UI: http://localhost:5000

## Azure Deployment

### Prerequisites

- Azure CLI installed
- Azure subscription
- GitHub repository

### Automated Setup

1. **Run the Azure setup script**
   ```bash
   chmod +x setup-azure.sh
   ./setup-azure.sh
   ```

2. **Configure GitHub Secrets**
   - `AZURE_CREDENTIALS`: Service principal JSON
   - `AZURE_WEBAPP_PUBLISH_PROFILE`: App Service publish profile

3. **Deploy via GitHub Actions**
   - Push to `main` branch triggers deployment
   - Monitor deployment in GitHub Actions tab

### Manual Setup

1. **Create Azure Resources**
   ```bash
   # Resource Group
   az group create --name expense-categorizer-rg --location eastus
   
   # App Service Plan (Free tier)
   az appservice plan create --name expense-categorizer-plan \
     --resource-group expense-categorizer-rg --location eastus --sku FREE --is-linux
   
   # Web App
   az webapp create --name expense-categorizer-prod \
     --resource-group expense-categorizer-rg --plan expense-categorizer-plan \
     --runtime "PYTHON|3.11"
   ```

2. **Configure App Settings**
   ```bash
   az webapp config appsettings set --name expense-categorizer-prod \
     --resource-group expense-categorizer-rg \
     --settings MLFLOW_TRACKING_URI="sqlite:///mlruns/mlflow.db"
   ```

## API Endpoints

### Core Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /categorize` - Categorize single transaction
- `POST /categorize/batch` - Categorize multiple transactions
- `POST /ingest` - Ingest from OneDrive
- `POST /process` - Process ingested files
- `POST /upload` - Upload transaction file
- `GET /categories` - Get available categories
- `GET /stats` - Get processing statistics

### Example Usage

```python
import requests

# Categorize a single transaction
response = requests.post("http://localhost:8000/categorize", json={
    "description": "Starbucks Coffee",
    "amount": 5.50
})
print(response.json())
# {"description": "Starbucks Coffee", "amount": 5.50, "category": "Food & Dining", "method": "rule_based"}
```

## Monitoring

### Health Checks

The application includes comprehensive monitoring:

- **Service Health**: Directory checks, MLflow connectivity
- **Data Quality**: Missing data, duplicates, distribution analysis
- **Model Performance**: Accuracy tracking, drift detection
- **System Resources**: Disk usage, file counts

### Monitoring Endpoints

- `GET /health` - Service health status
- `GET /stats` - Processing statistics
- Background monitoring via `src/monitoring.py`

### Alerting

Configure email alerts by setting environment variables:
- `SMTP_SERVER`
- `SMTP_PORT`
- `ALERT_EMAIL_FROM`
- `ALERT_EMAIL_TO`

## Development

### Running Tests

```bash
pytest tests/ -v --cov=src
```

### Code Quality

```bash
# Linting
flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics

# Formatting
black src/
```

### Prefect Pipeline

Run the complete pipeline:

```bash
python src/pipeline.py
```

## Free Azure Services Used

This project is optimized for free Microsoft Azure services:

- **Azure App Service** (Free Tier): Host the FastAPI application
- **Azure Storage** (Free Tier): Store transaction data and models
- **Microsoft Graph API** (Free): Access OneDrive data
- **Azure Key Vault** (Free Tier): Secure credential storage
- **GitHub Actions** (Free): CI/CD automation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details
