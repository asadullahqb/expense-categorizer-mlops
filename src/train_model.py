"""
ML Model Training Module

This module handles training and evaluation of ML models for transaction categorization.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    """Handles ML model training and evaluation"""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(random_state=42, probability=True)
        }
        
        self.best_model = None
        self.best_score = 0
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
    def load_training_data(self, data_path: str) -> pd.DataFrame:
        """Load training data with categories"""
        try:
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.json'):
                df = pd.read_json(data_path)
            else:
                raise ValueError("Unsupported file format")
            
            print(f"✅ Loaded training data: {len(df)} samples")
            return df
            
        except Exception as e:
            print(f"❌ Error loading training data: {str(e)}")
            raise
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for training"""
        # Combine description and amount as text
        df['combined_text'] = df['description'].astype(str) + ' ' + df['amount'].astype(str)
        
        # Vectorize text
        X = self.vectorizer.fit_transform(df['combined_text'])
        y = df['category'].values
        
        print(f"📊 Feature matrix shape: {X.shape}")
        print(f"📊 Target classes: {len(np.unique(y))}")
        
        return X, y
    
    def train_single_model(self, X: np.ndarray, y: np.ndarray, model_name: str, model) -> Dict:
        """Train a single model and return metrics"""
        print(f"\n🤖 Training {model_name}...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        print(f"✅ {model_name} - Accuracy: {accuracy:.3f}, CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return metrics
    
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray, model_name: str) -> Dict:
        """Perform hyperparameter tuning for the best model"""
        print(f"\n🔧 Hyperparameter tuning for {model_name}...")
        
        if model_name == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
            model = RandomForestClassifier(random_state=42)
            
        elif model_name == 'gradient_boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
            model = GradientBoostingClassifier(random_state=42)
            
        else:
            print(f"⚠️ No hyperparameter tuning defined for {model_name}")
            return {}
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X, y)
        
        print(f"✅ Best parameters: {grid_search.best_params_}")
        print(f"✅ Best score: {grid_search.best_score_:.3f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_model': grid_search.best_estimator_
        }
    
    def train_all_models(self, data_path: str) -> Dict:
        """Train all models and compare performance"""
        # Load data
        df = self.load_training_data(data_path)
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Train all models
        results = {}
        
        with mlflow.start_run(run_name="model_comparison"):
            for model_name, model in self.models.items():
                try:
                    metrics = self.train_single_model(X, y, model_name, model)
                    results[model_name] = metrics
                    
                    # Log to MLflow
                    mlflow.log_metric(f"{model_name}_accuracy", metrics['accuracy'])
                    mlflow.log_metric(f"{model_name}_cv_mean", metrics['cv_mean'])
                    mlflow.log_metric(f"{model_name}_cv_std", metrics['cv_std'])
                    
                except Exception as e:
                    print(f"❌ Error training {model_name}: {str(e)}")
                    continue
            
            # Find best model
            best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
            self.best_model = self.models[best_model_name]
            self.best_score = results[best_model_name]['accuracy']
            
            print(f"\n🏆 Best model: {best_model_name} with accuracy {self.best_score:.3f}")
            
            # Hyperparameter tuning for best model
            if best_model_name in ['random_forest', 'gradient_boosting']:
                tuning_results = self.hyperparameter_tuning(X, y, best_model_name)
                if tuning_results:
                    results[f"{best_model_name}_tuned"] = tuning_results
                    self.best_model = tuning_results['best_model']
                    self.best_score = tuning_results['best_score']
            
            # Log best model
            mlflow.sklearn.log_model(self.best_model, "best_model")
            mlflow.sklearn.log_model(self.vectorizer, "vectorizer")
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_metric("best_accuracy", self.best_score)
        
        return results
    
    def save_model(self, model_path: str = "models"):
        """Save the best trained model"""
        if self.best_model is None:
            print("⚠️ No trained model to save")
            return
        
        model_dir = Path(model_path)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and vectorizer
        joblib.dump(self.best_model, model_dir / "best_model.pkl")
        joblib.dump(self.vectorizer, model_dir / "vectorizer.pkl")
        
        # Save model metadata
        metadata = {
            'model_type': type(self.best_model).__name__,
            'accuracy': self.best_score,
            'features': self.vectorizer.get_feature_names_out().tolist() if hasattr(self.vectorizer, 'get_feature_names_out') else []
        }
        
        with open(model_dir / "model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Best model saved to {model_dir}")
    
    def create_visualizations(self, results: Dict, output_dir: str = "notebooks"):
        """Create visualization plots for model comparison"""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Model comparison plot
            model_names = list(results.keys())
            accuracies = [results[name]['accuracy'] for name in model_names]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(model_names, accuracies)
            plt.title('Model Performance Comparison')
            plt.xlabel('Models')
            plt.ylabel('Accuracy')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom')
            
            plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Confusion matrix for best model
            best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
            cm = np.array(results[best_model_name]['confusion_matrix'])
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {best_model_name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Visualizations saved to {output_dir}")
            
        except Exception as e:
            print(f"❌ Error creating visualizations: {str(e)}")

def main():
    """Main function for training models"""
    trainer = ModelTrainer()
    
    # Look for training data
    data_dir = Path("data/processed")
    if data_dir.exists():
        data_files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.json"))
        if data_files:
            # Use the most recent file
            latest_file = max(data_files, key=lambda f: f.stat().st_mtime)
            print(f"📁 Using training data: {latest_file}")
            
            # Train models
            results = trainer.train_all_models(str(latest_file))
            
            # Save best model
            trainer.save_model()
            
            # Create visualizations
            trainer.create_visualizations(results)
            
            print("\n📊 Training Results Summary:")
            for model_name, metrics in results.items():
                if isinstance(metrics, dict) and 'accuracy' in metrics:
                    print(f"  {model_name}: {metrics['accuracy']:.3f}")
        else:
            print("❌ No training data found in data/processed")
    else:
        print("❌ data/processed directory not found")

if __name__ == "__main__":
    main()

