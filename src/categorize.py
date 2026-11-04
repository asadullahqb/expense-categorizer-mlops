"""
Transaction Categorization Module

This module provides multiple categorization approaches:
1. Rule-based categorization using keywords
2. LLM-based categorization using OpenAI GPT
3. ML-based categorization using scikit-learn
"""

import pandas as pd
import numpy as np
import re
import os
from typing import List, Dict, Optional, Tuple
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TransactionCategorizer:
    """Handles transaction categorization using multiple approaches"""
    
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        
        # Define expense categories
        self.categories = {
            'Food & Dining': ['restaurant', 'cafe', 'food', 'dining', 'grocery', 'supermarket', 'coffee', 'lunch', 'dinner'],
            'Transportation': ['gas', 'fuel', 'uber', 'lyft', 'taxi', 'parking', 'toll', 'metro', 'bus', 'train'],
            'Shopping': ['amazon', 'store', 'shop', 'retail', 'clothing', 'shoes', 'electronics', 'online'],
            'Entertainment': ['movie', 'cinema', 'netflix', 'spotify', 'game', 'entertainment', 'theater', 'concert'],
            'Healthcare': ['doctor', 'hospital', 'pharmacy', 'medical', 'health', 'dental', 'clinic'],
            'Utilities': ['electric', 'water', 'internet', 'phone', 'utility', 'cable', 'internet'],
            'Education': ['school', 'university', 'course', 'education', 'book', 'tuition', 'learning'],
            'Travel': ['hotel', 'flight', 'airline', 'travel', 'vacation', 'trip', 'booking'],
            'Insurance': ['insurance', 'premium', 'coverage', 'policy'],
            'Investment': ['investment', 'stock', 'fund', 'portfolio', 'trading', 'brokerage'],
            'Transfer': ['transfer', 'deposit', 'withdrawal', 'atm', 'cash'],
            'Other': []  # Default category
        }
        
        # Initialize ML components
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
    
    def rule_based_categorization(self, description: str) -> str:
        """Categorize transaction using rule-based approach"""
        description_lower = description.lower()
        
        # Score each category based on keyword matches
        category_scores = {}
        for category, keywords in self.categories.items():
            score = 0
            for keyword in keywords:
                if keyword in description_lower:
                    score += 1
            category_scores[category] = score
        
        # Return category with highest score, default to 'Other'
        best_category = max(category_scores, key=category_scores.get)
        return best_category if category_scores[best_category] > 0 else 'Other'
    
    def llm_categorization(self, description: str, amount: float = None) -> str:
        """Categorize transaction using OpenAI GPT"""
        if not self.openai_api_key:
            print("⚠️ OpenAI API key not found, falling back to rule-based")
            return self.rule_based_categorization(description)
        
        try:
            # Create prompt for GPT
            prompt = f"""
            Categorize this transaction description into one of these categories:
            Food & Dining, Transportation, Shopping, Entertainment, Healthcare, 
            Utilities, Education, Travel, Insurance, Investment, Transfer, Other
            
            Transaction: {description}
            Amount: ${amount if amount else 'N/A'}
            
            Return only the category name, nothing else.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1
            )
            
            category = response.choices[0].message.content.strip()
            
            # Validate category
            if category in self.categories:
                return category
            else:
                print(f"⚠️ Invalid category from LLM: {category}, using rule-based")
                return self.rule_based_categorization(description)
                
        except Exception as e:
            print(f"❌ LLM categorization error: {str(e)}")
            return self.rule_based_categorization(description)
    
    def prepare_ml_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for ML model"""
        # Combine description and amount as text features
        df['combined_text'] = df['description'].astype(str) + ' ' + df['amount'].astype(str)
        
        # Vectorize text
        X = self.vectorizer.fit_transform(df['combined_text'])
        
        return X, df['category'].values if 'category' in df.columns else None
    
    def train_ml_model(self, df: pd.DataFrame) -> Dict:
        """Train ML model for categorization"""
        if 'category' not in df.columns:
            raise ValueError("Training data must have 'category' column")
        
        print("🤖 Training ML model...")
        
        # Prepare features
        X, y = self.prepare_ml_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.ml_model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate model
        y_pred = self.ml_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ ML model trained with {accuracy:.3f} accuracy")
        
        # Log to MLflow
        with mlflow.start_run(run_name="ml_model_training"):
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("n_estimators", 100)
            mlflow.sklearn.log_model(self.ml_model, "model")
            mlflow.sklearn.log_model(self.vectorizer, "vectorizer")
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def ml_categorization(self, description: str, amount: float = None) -> str:
        """Categorize transaction using trained ML model"""
        if not self.is_trained:
            print("⚠️ ML model not trained, using rule-based")
            return self.rule_based_categorization(description)
        
        try:
            # Prepare features
            combined_text = f"{description} {amount if amount else ''}"
            X = self.vectorizer.transform([combined_text])
            
            # Predict
            category = self.ml_model.predict(X)[0]
            return category
            
        except Exception as e:
            print(f"❌ ML categorization error: {str(e)}")
            return self.rule_based_categorization(description)
    
    def categorize_transactions(self, df: pd.DataFrame, method: str = 'rule_based') -> pd.DataFrame:
        """
        Categorize transactions using specified method
        
        Args:
            df: DataFrame with 'description' and 'amount' columns
            method: 'rule_based', 'llm', or 'ml'
        
        Returns:
            DataFrame with added 'category' column
        """
        df_categorized = df.copy()
        categories = []
        
        print(f"🏷️ Categorizing {len(df)} transactions using {method} method...")
        
        for idx, row in df.iterrows():
            description = str(row['description'])
            amount = row.get('amount', None)
            
            if method == 'rule_based':
                category = self.rule_based_categorization(description)
            elif method == 'llm':
                category = self.llm_categorization(description, amount)
            elif method == 'ml':
                category = self.ml_categorization(description, amount)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            categories.append(category)
        
        df_categorized['category'] = categories
        
        # Log categorization results
        category_counts = df_categorized['category'].value_counts()
        print(f"\n📊 Categorization Results:")
        for category, count in category_counts.items():
            print(f"  {category}: {count} transactions")
        
        return df_categorized
    
    def save_model(self, model_path: str = "models"):
        """Save trained model and vectorizer"""
        if not self.is_trained:
            print("⚠️ No trained model to save")
            return
        
        model_dir = Path(model_path)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and vectorizer
        joblib.dump(self.ml_model, model_dir / "categorizer_model.pkl")
        joblib.dump(self.vectorizer, model_dir / "vectorizer.pkl")
        
        # Save categories
        with open(model_dir / "categories.json", 'w') as f:
            json.dump(self.categories, f, indent=2)
        
        print(f"💾 Model saved to {model_dir}")
    
    def load_model(self, model_path: str = "models"):
        """Load trained model and vectorizer"""
        model_dir = Path(model_path)
        
        try:
            self.ml_model = joblib.load(model_dir / "categorizer_model.pkl")
            self.vectorizer = joblib.load(model_dir / "vectorizer.pkl")
            self.is_trained = True
            print(f"✅ Model loaded from {model_dir}")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            self.is_trained = False

def main():
    """Main function for testing the categorization module"""
    categorizer = TransactionCategorizer()
    
    # Example transactions
    sample_transactions = pd.DataFrame({
        'description': [
            'Starbucks Coffee',
            'Shell Gas Station',
            'Amazon Purchase',
            'Netflix Subscription',
            'Doctor Visit',
            'Electric Bill',
            'University Tuition',
            'Hotel Booking',
            'Insurance Premium',
            'Stock Purchase'
        ],
        'amount': [5.50, 45.20, 29.99, 15.99, 150.00, 85.30, 2500.00, 200.00, 120.00, 1000.00]
    })
    
    print("🧪 Testing categorization methods...")
    
    # Test rule-based
    print("\n1. Rule-based categorization:")
    df_rule = categorizer.categorize_transactions(sample_transactions, 'rule_based')
    print(df_rule[['description', 'category']].to_string(index=False))
    
    # Test LLM (if API key available)
    if categorizer.openai_api_key:
        print("\n2. LLM categorization:")
        df_llm = categorizer.categorize_transactions(sample_transactions, 'llm')
        print(df_llm[['description', 'category']].to_string(index=False))
    else:
        print("\n2. LLM categorization: Skipped (no API key)")
    
    # Test ML (would need training data)
    print("\n3. ML categorization: Requires training data")

if __name__ == "__main__":
    main()
