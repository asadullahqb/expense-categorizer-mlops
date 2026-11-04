"""
Tests for the preprocessing module
"""

import pytest
import pandas as pd
from datetime import datetime
from src.preprocess import TransactionPreprocessor

class TestTransactionPreprocessor:
    """Test cases for TransactionPreprocessor"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.preprocessor = TransactionPreprocessor()
        
        # Sample data with various formats
        self.sample_data = pd.DataFrame({
            'Transaction Date': ['2023-01-15', '2023-01-16', '2023-01-17'],
            'Description': ['Coffee Shop', 'Gas Station', 'Grocery Store'],
            'Amount': ['$5.50', '$45.20', '$29.99'],
            'Category': ['Food', 'Transport', 'Food']
        })
    
    def test_standardize_columns(self):
        """Test column standardization"""
        result_df = self.preprocessor.standardize_columns(self.sample_data)
        
        # Check that columns are standardized
        expected_columns = ['date', 'description', 'amount', 'category']
        for col in expected_columns:
            assert col in result_df.columns
    
    def test_parse_date_column(self):
        """Test date parsing"""
        df_with_dates = pd.DataFrame({
            'date': ['2023-01-15', '2023-01-16', '2023-01-17']
        })
        
        result_df = self.preprocessor.parse_date_column(df_with_dates)
        
        # Check that dates are parsed correctly
        assert pd.api.types.is_datetime64_any_dtype(result_df['date'])
        assert not result_df['date'].isna().any()
    
    def test_clean_amount_column(self):
        """Test amount cleaning"""
        df_with_amounts = pd.DataFrame({
            'amount': ['$5.50', '$45.20', '$29.99', 'invalid', '100.00']
        })
        
        result_df = self.preprocessor.clean_amount_column(df_with_amounts)
        
        # Check that amounts are cleaned and converted to numeric
        assert pd.api.types.is_numeric_dtype(result_df['amount'])
        # Should have 4 valid amounts (invalid should be NaN)
        assert result_df['amount'].notna().sum() == 4
    
    def test_clean_description_column(self):
        """Test description cleaning"""
        df_with_descriptions = pd.DataFrame({
            'description': ['  Coffee Shop  ', 'Gas Station', '', 'Grocery Store']
        })
        
        result_df = self.preprocessor.clean_description_column(df_with_descriptions)
        
        # Check that descriptions are cleaned
        assert not result_df['description'].str.contains('  ').any()  # No double spaces
        assert not (result_df['description'] == '').any()  # No empty strings
        assert not (result_df['description'] == 'nan').any()  # No 'nan' strings
    
    def test_remove_duplicates(self):
        """Test duplicate removal"""
        df_with_duplicates = pd.DataFrame({
            'description': ['Coffee', 'Gas', 'Coffee', 'Gas'],
            'amount': [5.50, 45.20, 5.50, 45.20]
        })
        
        result_df = self.preprocessor.remove_duplicates(df_with_duplicates)
        
        # Should have 2 unique rows
        assert len(result_df) == 2
    
    def test_filter_valid_transactions(self):
        """Test filtering of valid transactions"""
        df_with_invalid = pd.DataFrame({
            'date': [datetime.now(), datetime.now(), None, datetime.now()],
            'description': ['Coffee', 'Gas', 'Grocery', ''],
            'amount': [5.50, 45.20, 0, 29.99]
        })
        
        result_df = self.preprocessor.filter_valid_transactions(df_with_invalid)
        
        # Should have 2 valid transactions (Coffee and Gas)
        assert len(result_df) == 2
        assert not result_df['date'].isna().any()
        assert not (result_df['description'] == '').any()
        assert not (result_df['amount'] == 0).any()
    
    def test_process_file_mock(self):
        """Test file processing with mock data"""
        # Create a temporary CSV file
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.sample_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            result_df = self.preprocessor.process_file(temp_file)
            
            # Check that processing worked
            assert len(result_df) > 0
            assert 'date' in result_df.columns
            assert 'description' in result_df.columns
            assert 'amount' in result_df.columns
            
        finally:
            # Clean up
            os.unlink(temp_file)

