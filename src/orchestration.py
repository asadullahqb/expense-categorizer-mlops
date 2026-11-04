"""
Pipeline Orchestration with Prefect

This module orchestrates the entire expense categorization pipeline.
"""

from prefect import flow, task, get_run_logger
from prefect.task_runners import ConcurrentTaskRunner
from datetime import datetime, timedelta
from pathlib import Path
import mlflow
import pandas as pd
from typing import Dict, List, Optional

# Import our modules
from ingest import OneDriveIngestion
from preprocess import TransactionPreprocessor
from categorize import TransactionCategorizer
from train_model import ModelTrainer

@task
def ingest_data_task(folder_path: str = "/Finance/Transactions") -> Dict:
    """Task to ingest data from OneDrive"""
    logger = get_run_logger()
    logger.info(f"Starting data ingestion from {folder_path}")
    
    ingester = OneDriveIngestion()
    result = ingester.ingest_transaction_files(folder_path)
    
    logger.info(f"Ingestion complete: {result['files_downloaded']} files downloaded")
    return result

@task
def preprocess_data_task(file_paths: List[str]) -> str:
    """Task to preprocess transaction data"""
    logger = get_run_logger()
    logger.info(f"Starting data preprocessing for {len(file_paths)} files")
    
    preprocessor = TransactionPreprocessor()
    df = preprocessor.process_multiple_files(file_paths)
    
    if df.empty:
        raise ValueError("No valid transactions found after preprocessing")
    
    # Save processed data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/processed/transactions_{timestamp}.csv"
    preprocessor.save_processed_data(df, output_path)
    
    logger.info(f"Preprocessing complete: {len(df)} transactions saved to {output_path}")
    return output_path

@task
def categorize_transactions_task(processed_file: str, method: str = "rule_based") -> str:
    """Task to categorize transactions"""
    logger = get_run_logger()
    logger.info(f"Starting categorization using {method} method")
    
    # Load processed data
    df = pd.read_csv(processed_file)
    
    # Categorize transactions
    categorizer = TransactionCategorizer()
    categorized_df = categorizer.categorize_transactions(df, method=method)
    
    # Save categorized data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/processed/categorized_{timestamp}.csv"
    categorized_df.to_csv(output_path, index=False)
    
    # Log categorization results
    category_counts = categorized_df['category'].value_counts()
    logger.info(f"Categorization complete: {len(categorized_df)} transactions categorized")
    for category, count in category_counts.items():
        logger.info(f"  {category}: {count} transactions")
    
    return output_path

@task
def train_model_task(processed_file: str) -> Dict:
    """Task to train ML model"""
    logger = get_run_logger()
    logger.info("Starting ML model training")
    
    trainer = ModelTrainer()
    results = trainer.train_all_models(processed_file)
    
    # Save the best model
    trainer.save_model()
    
    logger.info(f"Model training complete. Best accuracy: {trainer.best_score:.3f}")
    return results

@task
def validate_pipeline_task(ingestion_result: Dict, processed_file: str, categorized_file: str) -> Dict:
    """Task to validate pipeline results"""
    logger = get_run_logger()
    logger.info("Starting pipeline validation")
    
    # Load and analyze results
    df_processed = pd.read_csv(processed_file)
    df_categorized = pd.read_csv(categorized_file)
    
    validation_results = {
        'ingestion': {
            'files_processed': ingestion_result['files_processed'],
            'files_downloaded': ingestion_result['files_downloaded'],
            'errors': len(ingestion_result['errors'])
        },
        'preprocessing': {
            'transactions_processed': len(df_processed),
            'columns': list(df_processed.columns)
        },
        'categorization': {
            'transactions_categorized': len(df_categorized),
            'categories': df_categorized['category'].nunique(),
            'category_distribution': df_categorized['category'].value_counts().to_dict()
        }
    }
    
    logger.info("Pipeline validation complete")
    logger.info(f"  Transactions processed: {validation_results['preprocessing']['transactions_processed']}")
    logger.info(f"  Categories found: {validation_results['categorization']['categories']}")
    
    return validation_results

@flow(task_runner=ConcurrentTaskRunner())
def expense_categorization_pipeline(
    folder_path: str = "/Finance/Transactions",
    categorization_method: str = "rule_based",
    train_model: bool = True
) -> Dict:
    """
    Main pipeline flow for expense categorization
    
    Args:
        folder_path: OneDrive folder path containing transaction files
        categorization_method: Method for categorization (rule_based, llm, ml)
        train_model: Whether to train ML model
    
    Returns:
        Dict with pipeline results and validation
    """
    logger = get_run_logger()
    logger.info("Starting expense categorization pipeline")
    
    # Set up MLflow tracking
    mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
    
    with mlflow.start_run(run_name=f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Step 1: Ingest data from OneDrive
        ingestion_result = ingest_data_task(folder_path)
        
        if ingestion_result['files_downloaded'] == 0:
            logger.warning("No files downloaded from OneDrive")
            return {"status": "no_data", "message": "No files found in OneDrive"}
        
        # Step 2: Preprocess data
        file_paths = ingestion_result['local_paths']
        processed_file = preprocess_data_task(file_paths)
        
        # Step 3: Categorize transactions
        categorized_file = categorize_transactions_task(processed_file, categorization_method)
        
        # Step 4: Train model (optional)
        model_results = None
        if train_model:
            model_results = train_model_task(processed_file)
        
        # Step 5: Validate pipeline
        validation_results = validate_pipeline_task(ingestion_result, processed_file, categorized_file)
        
        # Log pipeline metrics to MLflow
        mlflow.log_metric("files_processed", ingestion_result['files_processed'])
        mlflow.log_metric("files_downloaded", ingestion_result['files_downloaded'])
        mlflow.log_metric("transactions_processed", validation_results['preprocessing']['transactions_processed'])
        mlflow.log_metric("categories_found", validation_results['categorization']['categories'])
        
        if model_results:
            mlflow.log_metric("model_accuracy", model_results.get('best_accuracy', 0))
        
        logger.info("Pipeline execution complete")
        
        return {
            "status": "success",
            "ingestion": ingestion_result,
            "preprocessing": {"file": processed_file},
            "categorization": {"file": categorized_file, "method": categorization_method},
            "model_training": model_results,
            "validation": validation_results
        }

@flow
def scheduled_pipeline_flow():
    """Scheduled pipeline flow for regular execution"""
    logger = get_run_logger()
    logger.info("Starting scheduled pipeline execution")
    
    # Run the main pipeline
    result = expense_categorization_pipeline(
        folder_path="/Finance/Transactions",
        categorization_method="rule_based",
        train_model=False  # Don't retrain model on every run
    )
    
    logger.info(f"Scheduled pipeline complete: {result['status']}")
    return result

@flow
def retrain_model_flow():
    """Flow for retraining the ML model"""
    logger = get_run_logger()
    logger.info("Starting model retraining")
    
    # Find the most recent processed data
    processed_dir = Path("data/processed")
    if not processed_dir.exists():
        raise ValueError("No processed data found")
    
    data_files = list(processed_dir.glob("*.csv"))
    if not data_files:
        raise ValueError("No CSV files found in processed data")
    
    latest_file = max(data_files, key=lambda f: f.stat().st_mtime)
    logger.info(f"Using training data: {latest_file}")
    
    # Train model
    trainer = ModelTrainer()
    results = trainer.train_all_models(str(latest_file))
    trainer.save_model()
    
    logger.info(f"Model retraining complete. Best accuracy: {trainer.best_score:.3f}")
    return results

def main():
    """Main function for running the pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Expense Categorization Pipeline")
    parser.add_argument("--folder", default="/Finance/Transactions", help="OneDrive folder path")
    parser.add_argument("--method", default="rule_based", choices=["rule_based", "llm", "ml"], 
                       help="Categorization method")
    parser.add_argument("--train", action="store_true", help="Train ML model")
    parser.add_argument("--retrain", action="store_true", help="Retrain existing model")
    
    args = parser.parse_args()
    
    if args.retrain:
        # Retrain model only
        result = retrain_model_flow()
        print(f"Model retraining complete: {result}")
    else:
        # Run full pipeline
        result = expense_categorization_pipeline(
            folder_path=args.folder,
            categorization_method=args.method,
            train_model=args.train
        )
        print(f"Pipeline complete: {result}")

if __name__ == "__main__":
    main()

