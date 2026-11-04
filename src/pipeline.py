"""
Prefect Pipeline for Expense Categorizer

This module defines the complete MLOps pipeline using Prefect for orchestration.
"""

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional

# Import our modules
from ingest import OneDriveIngestion
from preprocess import TransactionPreprocessor
from categorize import TransactionCategorizer
from train_model import ModelTrainer

@task
def ingest_data(folder_path: str = "/Finance/Transactions") -> Dict:
    """Task to ingest data from OneDrive"""
    logger = get_run_logger()
    logger.info(f"Starting data ingestion from {folder_path}")
    
    ingester = OneDriveIngestion()
    result = ingester.ingest_transaction_files(folder_path)
    
    logger.info(f"Ingestion complete: {result['files_downloaded']} files downloaded")
    return result

@task
def preprocess_data(ingestion_result: Dict) -> str:
    """Task to preprocess ingested data"""
    logger = get_run_logger()
    logger.info("Starting data preprocessing")
    
    if not ingestion_result['local_paths']:
        logger.warning("No files to process")
        return None
    
    preprocessor = TransactionPreprocessor()
    df = preprocessor.process_multiple_files(ingestion_result['local_paths'])
    
    if df.empty:
        logger.error("No valid transactions found after preprocessing")
        return None
    
    # Save processed data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/processed/transactions_{timestamp}.csv"
    preprocessor.save_processed_data(df, output_path)
    
    logger.info(f"Preprocessing complete: {len(df)} transactions saved to {output_path}")
    return output_path

@task
def categorize_transactions(processed_file: str, method: str = "rule_based") -> str:
    """Task to categorize transactions"""
    logger = get_run_logger()
    logger.info(f"Starting transaction categorization using {method} method")
    
    if not processed_file:
        logger.error("No processed file to categorize")
        return None
    
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
def train_ml_model(processed_file: str) -> Dict:
    """Task to train ML model"""
    logger = get_run_logger()
    logger.info("Starting ML model training")
    
    if not processed_file:
        logger.error("No processed file for training")
        return None
    
    trainer = ModelTrainer()
    results = trainer.train_all_models(processed_file)
    
    # Save the best model
    trainer.save_model()
    
    logger.info(f"ML training complete: Best accuracy {trainer.best_score:.3f}")
    return results

@task
def generate_report(categorized_file: str) -> Dict:
    """Task to generate analysis report"""
    logger = get_run_logger()
    logger.info("Generating analysis report")
    
    if not categorized_file:
        logger.error("No categorized file for report")
        return None
    
    # Load categorized data
    df = pd.read_csv(categorized_file)
    
    # Generate report
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_transactions': len(df),
        'total_amount': float(df['amount'].sum()) if 'amount' in df.columns else 0,
        'categories': df['category'].value_counts().to_dict() if 'category' in df.columns else {},
        'date_range': {
            'start': df['date'].min() if 'date' in df.columns else None,
            'end': df['date'].max() if 'date' in df.columns else None
        }
    }
    
    # Save report
    report_path = f"data/processed/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    import json
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Report generated: {report_path}")
    return report

@flow(task_runner=SequentialTaskRunner())
def expense_categorizer_pipeline(
    folder_path: str = "/Finance/Transactions",
    categorization_method: str = "rule_based",
    train_model: bool = True
) -> Dict:
    """
    Main pipeline for expense categorization
    
    Args:
        folder_path: OneDrive folder path containing transaction files
        categorization_method: Method for categorization (rule_based, llm, ml)
        train_model: Whether to train ML model
    
    Returns:
        Dict with pipeline results
    """
    logger = get_run_logger()
    logger.info("Starting Expense Categorizer Pipeline")
    
    # Step 1: Ingest data from OneDrive
    ingestion_result = ingest_data(folder_path)
    
    if not ingestion_result['files_downloaded']:
        logger.warning("No files downloaded, ending pipeline")
        return {'status': 'no_data', 'message': 'No files found to process'}
    
    # Step 2: Preprocess data
    processed_file = preprocess_data(ingestion_result)
    
    if not processed_file:
        logger.error("Preprocessing failed, ending pipeline")
        return {'status': 'preprocessing_failed', 'message': 'No valid transactions found'}
    
    # Step 3: Categorize transactions
    categorized_file = categorize_transactions(processed_file, categorization_method)
    
    if not categorized_file:
        logger.error("Categorization failed, ending pipeline")
        return {'status': 'categorization_failed', 'message': 'Categorization failed'}
    
    # Step 4: Train ML model (optional)
    ml_results = None
    if train_model:
        ml_results = train_ml_model(processed_file)
    
    # Step 5: Generate report
    report = generate_report(categorized_file)
    
    # Return pipeline results
    results = {
        'status': 'success',
        'ingestion': ingestion_result,
        'processed_file': processed_file,
        'categorized_file': categorized_file,
        'ml_results': ml_results,
        'report': report,
        'timestamp': datetime.now().isoformat()
    }
    
    logger.info("Expense Categorizer Pipeline completed successfully")
    return results

@flow
def scheduled_pipeline():
    """Scheduled pipeline that runs daily"""
    logger = get_run_logger()
    logger.info("Running scheduled expense categorizer pipeline")
    
    # Run the main pipeline
    results = expense_categorizer_pipeline(
        folder_path="/Finance/Transactions",
        categorization_method="rule_based",
        train_model=False  # Don't retrain model daily
    )
    
    return results

if __name__ == "__main__":
    # Run the pipeline
    results = expense_categorizer_pipeline()
    print(f"Pipeline results: {results}")

