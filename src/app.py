"""
FastAPI Application for Expense Categorizer

This module provides a REST API for the expense categorization service.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import json
from pathlib import Path
import asyncio
from datetime import datetime
import uvicorn

# Import our modules
from ingest import OneDriveIngestion
from preprocess import TransactionPreprocessor
from categorize import TransactionCategorizer

# Initialize FastAPI app
app = FastAPI(
    title="Expense Categorizer API",
    description="ML-powered expense categorization service with OneDrive integration",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
ingester = OneDriveIngestion()
preprocessor = TransactionPreprocessor()
categorizer = TransactionCategorizer()

# Pydantic models
class TransactionRequest(BaseModel):
    description: str
    amount: float
    date: Optional[str] = None

class TransactionResponse(BaseModel):
    description: str
    amount: float
    category: str
    confidence: Optional[float] = None
    method: str

class CategorizationRequest(BaseModel):
    transactions: List[TransactionRequest]
    method: str = "rule_based"  # rule_based, llm, ml

class CategorizationResponse(BaseModel):
    transactions: List[TransactionResponse]
    method: str
    total_processed: int

class IngestionResponse(BaseModel):
    status: str
    files_processed: int
    files_downloaded: int
    timestamp: str
    errors: List[str]

# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Expense Categorizer API",
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": {
            "health": "/health",
            "categorize": "/categorize",
            "categorize_batch": "/categorize/batch",
            "ingest": "/ingest",
            "process": "/process"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "ingestion": "ready",
            "preprocessing": "ready",
            "categorization": "ready"
        }
    }

@app.post("/categorize", response_model=TransactionResponse)
async def categorize_single_transaction(request: TransactionRequest):
    """Categorize a single transaction"""
    try:
        # Create DataFrame for single transaction
        df = pd.DataFrame({
            'description': [request.description],
            'amount': [request.amount],
            'date': [request.date or datetime.now().isoformat()]
        })
        
        # Categorize using rule-based method (fastest)
        categorized_df = categorizer.categorize_transactions(df, method='rule_based')
        
        return TransactionResponse(
            description=request.description,
            amount=request.amount,
            category=categorized_df['category'].iloc[0],
            method='rule_based'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/categorize/batch", response_model=CategorizationResponse)
async def categorize_batch_transactions(request: CategorizationRequest):
    """Categorize multiple transactions"""
    try:
        # Convert to DataFrame
        transactions_data = []
        for tx in request.transactions:
            transactions_data.append({
                'description': tx.description,
                'amount': tx.amount,
                'date': tx.date or datetime.now().isoformat()
            })
        
        df = pd.DataFrame(transactions_data)
        
        # Categorize transactions
        categorized_df = categorizer.categorize_transactions(df, method=request.method)
        
        # Convert to response format
        response_transactions = []
        for _, row in categorized_df.iterrows():
            response_transactions.append(TransactionResponse(
                description=row['description'],
                amount=row['amount'],
                category=row['category'],
                method=request.method
            ))
        
        return CategorizationResponse(
            transactions=response_transactions,
            method=request.method,
            total_processed=len(response_transactions)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest", response_model=IngestionResponse)
async def ingest_from_onedrive(folder_path: str = "/Finance/Transactions"):
    """Ingest transaction files from OneDrive"""
    try:
        result = ingester.ingest_transaction_files(folder_path)
        
        return IngestionResponse(
            status="success" if result['files_downloaded'] > 0 else "no_files",
            files_processed=result['files_processed'],
            files_downloaded=result['files_downloaded'],
            timestamp=result['timestamp'],
            errors=result['errors']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process")
async def process_ingested_files():
    """Process ingested files and return summary"""
    try:
        # Get all raw files
        raw_data_dir = Path("data/raw")
        if not raw_data_dir.exists():
            raise HTTPException(status_code=404, detail="No raw data directory found")
        
        file_paths = list(raw_data_dir.glob("*"))
        if not file_paths:
            raise HTTPException(status_code=404, detail="No files found in raw data directory")
        
        # Process files
        df = preprocessor.process_multiple_files([str(f) for f in file_paths])
        
        if df.empty:
            raise HTTPException(status_code=400, detail="No valid transactions found")
        
        # Save processed data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/processed/transactions_{timestamp}.csv"
        preprocessor.save_processed_data(df, output_path)
        
        return {
            "status": "success",
            "transactions_processed": len(df),
            "output_file": output_path,
            "date_range": {
                "start": df['date'].min().isoformat() if 'date' in df.columns else None,
                "end": df['date'].max().isoformat() if 'date' in df.columns else None
            },
            "amount_summary": {
                "total": float(df['amount'].sum()) if 'amount' in df.columns else 0,
                "average": float(df['amount'].mean()) if 'amount' in df.columns else 0,
                "min": float(df['amount'].min()) if 'amount' in df.columns else 0,
                "max": float(df['amount'].max()) if 'amount' in df.columns else 0
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_transaction_file(file: UploadFile = File(...)):
    """Upload and process a transaction file"""
    try:
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = f"data/raw/{filename}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process file
        df = preprocessor.process_file(file_path)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="No valid transactions found in file")
        
        # Categorize transactions
        categorized_df = categorizer.categorize_transactions(df, method='rule_based')
        
        # Save processed data
        output_path = f"data/processed/transactions_{timestamp}.csv"
        preprocessor.save_processed_data(categorized_df, output_path)
        
        return {
            "status": "success",
            "filename": filename,
            "transactions_processed": len(categorized_df),
            "output_file": output_path,
            "categories": categorized_df['category'].value_counts().to_dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/categories")
async def get_categories():
    """Get available expense categories"""
    return {
        "categories": list(categorizer.categories.keys()),
        "category_keywords": categorizer.categories
    }

@app.get("/stats")
async def get_statistics():
    """Get processing statistics"""
    try:
        # Count files in directories
        raw_count = len(list(Path("data/raw").glob("*"))) if Path("data/raw").exists() else 0
        processed_count = len(list(Path("data/processed").glob("*"))) if Path("data/processed").exists() else 0
        
        return {
            "raw_files": raw_count,
            "processed_files": processed_count,
            "mlflow_runs": len(list(Path("mlruns").glob("*"))) if Path("mlruns").exists() else 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Background task for model training
@app.post("/train")
async def train_model(background_tasks: BackgroundTasks):
    """Train ML model in background"""
    try:
        from train_model import ModelTrainer
        
        def train_background():
            trainer = ModelTrainer()
            # Look for training data
            data_dir = Path("data/processed")
            if data_dir.exists():
                data_files = list(data_dir.glob("*.csv"))
                if data_files:
                    latest_file = max(data_files, key=lambda f: f.stat().st_mtime)
                    results = trainer.train_all_models(str(latest_file))
                    trainer.save_model()
        
        background_tasks.add_task(train_background)
        
        return {
            "status": "training_started",
            "message": "Model training started in background"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

