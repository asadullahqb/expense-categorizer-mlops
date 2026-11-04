"""
Data Preprocessing Module

This module handles cleaning, standardizing, and preparing transaction data for categorization.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import re
from pathlib import Path
import mlflow
import json

class TransactionPreprocessor:
    """Handles cleaning and standardizing transaction data"""
    
    def __init__(self):
        self.standard_columns = ['date', 'description', 'amount', 'category']
        self.date_formats = [
            '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S',
            '%m-%d-%Y', '%d-%m-%Y', '%Y/%m/%d', '%m/%d/%y', '%d/%m/%y'
        ]
    
    def detect_file_format(self, file_path: str) -> str:
        """Detect file format and return appropriate reader"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.csv':
            return 'csv'
        elif file_ext in ['.xlsx', '.xls']:
            return 'excel'
        elif file_ext == '.json':
            return 'json'
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def load_transaction_file(self, file_path: str) -> pd.DataFrame:
        """Load transaction file and return DataFrame"""
        try:
            file_format = self.detect_file_format(file_path)
            
            if file_format == 'csv':
                # Try different encodings and separators
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    for sep in [',', ';', '\t']:
                        try:
                            df = pd.read_csv(file_path, encoding=encoding, sep=sep)
                            if len(df.columns) > 1:  # Valid CSV
                                print(f"✅ Loaded CSV with {len(df)} rows using {encoding} encoding")
                                return df
                        except:
                            continue
                raise ValueError("Could not parse CSV file")
                
            elif file_format == 'excel':
                df = pd.read_excel(file_path)
                print(f"✅ Loaded Excel with {len(df)} rows")
                return df
                
            elif file_format == 'json':
                df = pd.read_json(file_path)
                print(f"✅ Loaded JSON with {len(df)} rows")
                return df
                
        except Exception as e:
            print(f"❌ Error loading {file_path}: {str(e)}")
            raise
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to our schema"""
        # Common column name mappings
        column_mappings = {
            # Date columns
            'transaction_date': 'date', 'txn_date': 'date', 'date_time': 'date',
            'transaction_time': 'date', 'posted_date': 'date', 'value_date': 'date',
            
            # Description columns
            'transaction_description': 'description', 'txn_description': 'description',
            'description': 'description', 'memo': 'description', 'details': 'description',
            'payee': 'description', 'merchant': 'description', 'reference': 'description',
            
            # Amount columns
            'transaction_amount': 'amount', 'txn_amount': 'amount', 'amount': 'amount',
            'debit': 'amount', 'credit': 'amount', 'value': 'amount',
            
            # Category columns
            'category': 'category', 'classification': 'category', 'type': 'category'
        }
        
        # Create a copy to avoid modifying original
        df_std = df.copy()
        
        # Rename columns based on mappings
        df_std.columns = df_std.columns.str.lower().str.strip()
        df_std = df_std.rename(columns=column_mappings)
        
        print(f"📊 Standardized columns: {list(df_std.columns)}")
        return df_std
    
    def parse_date_column(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """Parse and standardize date column"""
        if date_col not in df.columns:
            print(f"⚠️ Date column '{date_col}' not found")
            return df
        
        df_parsed = df.copy()
        
        # Try different date formats
        for date_format in self.date_formats:
            try:
                df_parsed[date_col] = pd.to_datetime(df_parsed[date_col], format=date_format, errors='coerce')
                if not df_parsed[date_col].isna().all():
                    print(f"✅ Parsed dates using format: {date_format}")
                    break
            except:
                continue
        
        # If all formats failed, try pandas auto-parsing
        if df_parsed[date_col].isna().all():
            try:
                df_parsed[date_col] = pd.to_datetime(df_parsed[date_col], infer_datetime_format=True)
                print("✅ Parsed dates using auto-inference")
            except:
                print("❌ Could not parse date column")
        
        return df_parsed
    
    def clean_amount_column(self, df: pd.DataFrame, amount_col: str = 'amount') -> pd.DataFrame:
        """Clean and standardize amount column"""
        if amount_col not in df.columns:
            print(f"⚠️ Amount column '{amount_col}' not found")
            return df
        
        df_cleaned = df.copy()
        
        # Convert to string first to handle mixed types
        df_cleaned[amount_col] = df_cleaned[amount_col].astype(str)
        
        # Remove currency symbols and clean
        df_cleaned[amount_col] = df_cleaned[amount_col].str.replace(r'[^\d.-]', '', regex=True)
        
        # Handle negative amounts (debits)
        df_cleaned[amount_col] = df_cleaned[amount_col].str.replace('--', '-')
        
        # Convert to numeric
        df_cleaned[amount_col] = pd.to_numeric(df_cleaned[amount_col], errors='coerce')
        
        print(f"✅ Cleaned amount column: {df_cleaned[amount_col].notna().sum()} valid amounts")
        return df_cleaned
    
    def clean_description_column(self, df: pd.DataFrame, desc_col: str = 'description') -> pd.DataFrame:
        """Clean and standardize description column"""
        if desc_col not in df.columns:
            print(f"⚠️ Description column '{desc_col}' not found")
            return df
        
        df_cleaned = df.copy()
        
        # Convert to string and clean
        df_cleaned[desc_col] = df_cleaned[desc_col].astype(str)
        df_cleaned[desc_col] = df_cleaned[desc_col].str.strip()
        df_cleaned[desc_col] = df_cleaned[desc_col].str.replace(r'\s+', ' ', regex=True)  # Multiple spaces to single
        
        # Remove empty descriptions
        df_cleaned = df_cleaned[df_cleaned[desc_col] != '']
        df_cleaned = df_cleaned[df_cleaned[desc_col] != 'nan']
        
        print(f"✅ Cleaned description column: {len(df_cleaned)} valid descriptions")
        return df_cleaned
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate transactions"""
        initial_count = len(df)
        df_deduped = df.drop_duplicates()
        removed_count = initial_count - len(df_deduped)
        
        if removed_count > 0:
            print(f"🗑️ Removed {removed_count} duplicate transactions")
        
        return df_deduped
    
    def filter_valid_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out invalid transactions"""
        initial_count = len(df)
        
        # Remove rows with missing essential data
        df_filtered = df.dropna(subset=['date', 'description', 'amount'])
        
        # Remove zero amounts (unless they're legitimate)
        df_filtered = df_filtered[df_filtered['amount'] != 0]
        
        # Remove very old transactions (older than 10 years)
        if 'date' in df_filtered.columns:
            cutoff_date = datetime.now().replace(year=datetime.now().year - 10)
            df_filtered = df_filtered[df_filtered['date'] >= cutoff_date]
        
        removed_count = initial_count - len(df_filtered)
        if removed_count > 0:
            print(f"🔍 Filtered out {removed_count} invalid transactions")
        
        return df_filtered
    
    def process_file(self, file_path: str) -> pd.DataFrame:
        """Process a single transaction file"""
        print(f"\n🔄 Processing: {Path(file_path).name}")
        
        # Load file
        df = self.load_transaction_file(file_path)
        
        # Standardize columns
        df = self.standardize_columns(df)
        
        # Parse dates
        df = self.parse_date_column(df)
        
        # Clean amounts
        df = self.clean_amount_column(df)
        
        # Clean descriptions
        df = self.clean_description_column(df)
        
        # Remove duplicates
        df = self.remove_duplicates(df)
        
        # Filter valid transactions
        df = self.filter_valid_transactions(df)
        
        print(f"✅ Processed: {len(df)} valid transactions")
        return df
    
    def process_multiple_files(self, file_paths: List[str]) -> pd.DataFrame:
        """Process multiple transaction files and combine them"""
        all_transactions = []
        
        for file_path in file_paths:
            try:
                df = self.process_file(file_path)
                if not df.empty:
                    # Add source file information
                    df['source_file'] = Path(file_path).name
                    all_transactions.append(df)
            except Exception as e:
                print(f"❌ Error processing {file_path}: {str(e)}")
                continue
        
        if not all_transactions:
            print("❌ No valid transactions found")
            return pd.DataFrame()
        
        # Combine all transactions
        combined_df = pd.concat(all_transactions, ignore_index=True)
        
        # Final cleanup
        combined_df = self.remove_duplicates(combined_df)
        combined_df = combined_df.sort_values('date', ascending=False)
        
        print(f"\n📊 Combined dataset: {len(combined_df)} transactions from {len(all_transactions)} files")
        
        return combined_df
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str) -> bool:
        """Save processed data to file"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if output_path.suffix.lower() == '.csv':
                df.to_csv(output_path, index=False)
            elif output_path.suffix.lower() in ['.xlsx', '.xls']:
                df.to_excel(output_path, index=False)
            elif output_path.suffix.lower() == '.json':
                df.to_json(output_path, orient='records', date_format='iso')
            else:
                # Default to CSV
                output_path = output_path.with_suffix('.csv')
                df.to_csv(output_path, index=False)
            
            print(f"💾 Saved processed data: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving data: {str(e)}")
            return False

def main():
    """Main function for testing the preprocessing module"""
    preprocessor = TransactionPreprocessor()
    
    # Example usage
    raw_data_dir = Path("data/raw")
    if raw_data_dir.exists():
        file_paths = list(raw_data_dir.glob("*"))
        if file_paths:
            print(f"Found {len(file_paths)} files to process")
            df = preprocessor.process_multiple_files([str(f) for f in file_paths])
            
            if not df.empty:
                # Save processed data
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"data/processed/transactions_{timestamp}.csv"
                preprocessor.save_processed_data(df, output_path)
                
                print(f"\n📈 Data Summary:")
                print(f"Total transactions: {len(df)}")
                print(f"Date range: {df['date'].min()} to {df['date'].max()}")
                print(f"Amount range: ${df['amount'].min():.2f} to ${df['amount'].max():.2f}")
        else:
            print("No files found in data/raw directory")
    else:
        print("data/raw directory not found")

if __name__ == "__main__":
    main()

