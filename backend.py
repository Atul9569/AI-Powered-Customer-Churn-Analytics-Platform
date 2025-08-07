"""
Backend processing module for ChurnAI Platform
Handles data processing, model training, and predictions
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import core modules
from src.multi_dataset_loader import MultiDatasetLoader
from src.preprocessing import ChurnPreprocessor

class ChurnBackend:
    """Backend service for churn prediction platform"""
    
    def __init__(self):
        self.dataset_loader = MultiDatasetLoader()
        self.preprocessor = ChurnPreprocessor()
        self.models = {}
        self.training_results = {}
        
    def get_available_datasets(self) -> Dict[str, Any]:
        """Get information about available datasets"""
        return {
            "Telecommunications": {
                "icon": "📱",
                "description": "Telecom customer churn analysis",
                "customers": 7043,
                "features": 19
            },
            "Banking": {
                "icon": "🏦", 
                "description": "Banking customer retention",
                "customers": 5000,
                "features": 15
            },
            "E-commerce": {
                "icon": "🛒",
                "description": "Online retail customer behavior", 
                "customers": 4500,
                "features": 12
            }
        }
    
    def load_dataset(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """Load a specific dataset"""
        return self.dataset_loader.load_dataset(dataset_name)
    
    def get_dataset_stats(self, dataset_name: str) -> Dict[str, Any]:
        """Get statistical information about a dataset"""
        data = self.load_dataset(dataset_name)
        if data is None:
            return {}
        
        stats = {
            'total_customers': len(data),
            'features': len(data.columns) - 1,  # Excluding target
            'churn_rate': 0,
            'numerical_features': len(data.select_dtypes(include=['int64', 'float64']).columns),
            'categorical_features': len(data.select_dtypes(include=['object']).columns),
            'missing_values': data.isnull().sum().sum()
        }
        
        if 'Churn' in data.columns:
            stats['churn_rate'] = (data['Churn'].sum() / len(data) * 100)
        
        return stats
    
    def train_models(self, dataset_name: str = "Telecommunications") -> Dict[str, Any]:
        """Train all models on specified dataset"""
        try:
            # Load and prepare data
            data = self.load_dataset(dataset_name)
            if data is None:
                return {"error": "Failed to load dataset"}
            
            # Prepare features and target
            if 'Churn' in data.columns:
                X = data.drop('Churn', axis=1)
                y = data['Churn']
            else:
                return {"error": "No target column 'Churn' found"}
            
            # Preprocess data
            X_processed = self.preprocessor.fit_transform(X)
            
            # Train models
            models = {
                'logistic_regression': 'LogisticRegression',
                'random_forest': 'RandomForestClassifier', 
                'xgboost': 'XGBClassifier'
            }
            
            results = {}
            
            for model_name, model_class in models.items():
                print(f"Training {model_name}...")
                
                # Train model
                trained_model, training_results = self.model_trainer.train_model(
                    X_processed, y, model_class, model_name
                )
                
                if trained_model:
                    self.models[model_name] = trained_model
                    results[model_name] = training_results
                    
                    # Save model
                    os.makedirs('models', exist_ok=True)
                    joblib.dump(trained_model, f'models/{model_name}_model.joblib')
            
            # Save preprocessor and results
            joblib.dump(self.preprocessor, 'models/preprocessor.joblib')
            joblib.dump(results, 'models/training_results.joblib')
            
            self.training_results = results
            return results
            
        except Exception as e:
            return {"error": f"Training failed: {str(e)}"}
    
    def load_trained_models(self) -> bool:
        """Load previously trained models"""
        try:
            model_files = {
                'logistic_regression': 'models/logistic_regression_model.joblib',
                'random_forest': 'models/random_forest_model.joblib',
                'xgboost': 'models/xgboost_model.joblib'
            }
            
            for model_name, file_path in model_files.items():
                if os.path.exists(file_path):
                    self.models[model_name] = joblib.load(file_path)
            
            # Load preprocessor
            if os.path.exists('models/preprocessor.joblib'):
                self.preprocessor = joblib.load('models/preprocessor.joblib')
            
            # Load training results
            if os.path.exists('models/training_results.joblib'):
                self.training_results = joblib.load('models/training_results.joblib')
            
            return len(self.models) > 0
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def predict_churn(self, customer_data: Dict[str, Any], model_name: str = 'xgboost') -> Dict[str, Any]:
        """Make churn prediction for a customer"""
        try:
            # Convert input to DataFrame
            input_df = pd.DataFrame([customer_data])
            
            # Load model if not already loaded
            if model_name not in self.models:
                if not self.load_trained_models():
                    return {"error": "No trained models available"}
            
            # Preprocess input
            if not hasattr(self.preprocessor, 'fitted_encoders'):
                if os.path.exists('models/preprocessor.joblib'):
                    self.preprocessor = joblib.load('models/preprocessor.joblib')
                else:
                    return {"error": "Preprocessor not fitted"}
            
            X_processed = self.preprocessor.transform(input_df)
            
            # Make prediction
            model = self.models[model_name]
            prediction = model.predict(X_processed)[0]
            probabilities = model.predict_proba(X_processed)[0]
            
            return {
                'prediction': int(prediction),
                'churn_probability': float(probabilities[1]),
                'retention_probability': float(probabilities[0]),
                'model_used': model_name,
                'confidence': float(max(probabilities))
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all trained models"""
        if os.path.exists('models/training_results.joblib'):
            return joblib.load('models/training_results.joblib')
        return self.training_results
    
    def get_feature_importance(self, model_name: str = 'xgboost') -> Dict[str, float]:
        """Get feature importance for specified model"""
        try:
            if model_name not in self.models:
                self.load_trained_models()
            
            model = self.models.get(model_name)
            if not model:
                return {}
            
            # Get feature names from preprocessor
            feature_names = getattr(self.preprocessor, 'feature_names', [])
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                return dict(zip(feature_names[:len(importances)], importances))
            elif hasattr(model, 'coef_'):
                # For logistic regression
                coefficients = np.abs(model.coef_[0])
                return dict(zip(feature_names[:len(coefficients)], coefficients))
            
            return {}
            
        except Exception as e:
            print(f"Error getting feature importance: {e}")
            return {}
    
    def generate_business_insights(self, dataset_name: str = "Telecommunications") -> Dict[str, Any]:
        """Generate business insights from the data and models"""
        try:
            data = self.load_dataset(dataset_name)
            if data is None:
                return {}
            
            insights = {
                'dataset_overview': self.get_dataset_stats(dataset_name),
                'churn_patterns': {},
                'recommendations': []
            }
            
            # Analyze churn patterns
            if 'Churn' in data.columns:
                churned = data[data['Churn'] == 1]
                retained = data[data['Churn'] == 0]
                
                # Monthly charges comparison
                if 'MonthlyCharges' in data.columns:
                    insights['churn_patterns']['avg_monthly_charges'] = {
                        'churned': float(churned['MonthlyCharges'].mean()),
                        'retained': float(retained['MonthlyCharges'].mean())
                    }
                
                # Contract type analysis
                if 'Contract' in data.columns:
                    contract_churn = data.groupby('Contract')['Churn'].mean().to_dict()
                    insights['churn_patterns']['contract_churn_rates'] = contract_churn
                
                # Generate recommendations
                if insights['churn_patterns'].get('avg_monthly_charges'):
                    if insights['churn_patterns']['avg_monthly_charges']['churned'] > \
                       insights['churn_patterns']['avg_monthly_charges']['retained']:
                        insights['recommendations'].append(
                            "High monthly charges correlate with churn. Consider pricing optimization."
                        )
                
                if contract_churn and 'Month-to-month' in contract_churn:
                    if contract_churn['Month-to-month'] > 0.4:
                        insights['recommendations'].append(
                            "Month-to-month customers show high churn. Offer incentives for longer contracts."
                        )
            
            return insights
            
        except Exception as e:
            return {"error": f"Failed to generate insights: {str(e)}"}

# Global backend instance
churn_backend = ChurnBackend()

def get_backend() -> ChurnBackend:
    """Get the global backend instance"""
    return churn_backend