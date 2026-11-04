"""
Tests for the categorization module
"""

import pytest
import pandas as pd
from src.categorize import TransactionCategorizer

class TestTransactionCategorizer:
    """Test cases for TransactionCategorizer"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.categorizer = TransactionCategorizer()
        self.sample_transactions = pd.DataFrame({
            'description': [
                'Starbucks Coffee',
                'Shell Gas Station',
                'Amazon Purchase',
                'Netflix Subscription',
                'Doctor Visit'
            ],
            'amount': [5.50, 45.20, 29.99, 15.99, 150.00]
        })
    
    def test_rule_based_categorization(self):
        """Test rule-based categorization"""
        # Test individual transactions
        assert self.categorizer.rule_based_categorization('Starbucks Coffee') == 'Food & Dining'
        assert self.categorizer.rule_based_categorization('Shell Gas Station') == 'Transportation'
        assert self.categorizer.rule_based_categorization('Amazon Purchase') == 'Shopping'
        assert self.categorizer.rule_based_categorization('Netflix Subscription') == 'Entertainment'
        assert self.categorizer.rule_based_categorization('Doctor Visit') == 'Healthcare'
    
    def test_categorize_transactions_rule_based(self):
        """Test batch categorization with rule-based method"""
        result_df = self.categorizer.categorize_transactions(
            self.sample_transactions, 
            method='rule_based'
        )
        
        # Check that all transactions have categories
        assert 'category' in result_df.columns
        assert len(result_df) == len(self.sample_transactions)
        assert not result_df['category'].isna().any()
        
        # Check specific categorizations
        expected_categories = [
            'Food & Dining',
            'Transportation', 
            'Shopping',
            'Entertainment',
            'Healthcare'
        ]
        
        for i, expected in enumerate(expected_categories):
            assert result_df.iloc[i]['category'] == expected
    
    def test_categorize_transactions_ml(self):
        """Test ML categorization (requires training)"""
        # This test will use rule-based fallback since no training data
        result_df = self.categorizer.categorize_transactions(
            self.sample_transactions,
            method='ml'
        )
        
        assert 'category' in result_df.columns
        assert len(result_df) == len(self.sample_transactions)
        assert not result_df['category'].isna().any()
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        empty_df = pd.DataFrame(columns=['description', 'amount'])
        result_df = self.categorizer.categorize_transactions(empty_df, method='rule_based')
        
        assert len(result_df) == 0
        assert 'category' in result_df.columns
    
    def test_invalid_method(self):
        """Test handling of invalid categorization method"""
        with pytest.raises(ValueError):
            self.categorizer.categorize_transactions(
                self.sample_transactions,
                method='invalid_method'
            )

