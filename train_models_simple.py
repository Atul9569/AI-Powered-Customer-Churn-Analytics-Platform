#!/usr/bin/env python3
"""
Simplified model training script for Customer Churn Prediction
Creates trained models quickly with basic parameters
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib
import os
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_simple_models():
    """Train models with basic parameters"""
    logger.info("Starting simplified model training...")
    
    try:
        # Load and prepare data
        from src.data_loader import TelcoDataLoader
        from src.preprocessing import prepare_data
        
        # Load data
        loader = TelcoDataLoader()
        data = loader.load_data()
        
        if data is None:
            logger.error("Failed to load data")
            return False
        
        # Split features and target
        X, y = loader.split_features_target(data)
        
        # Prepare data with preprocessing
        X_train, X_test, y_train, y_test, preprocessor = prepare_data(X, y, test_size=0.2, random_state=42)
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Define simple models
        models = {
            'logistic_regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                solver='lbfgs'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'xgboost': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss',
                verbosity=0
            )
        }
        
        # Train models
        trained_models = {}
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            try:
                # Train the model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                logger.info(f"{name} - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
                
                trained_models[name] = model
                results[name] = {
                    'accuracy': accuracy,
                    'roc_auc': roc_auc
                }
                
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                continue
        
        # Save models
        os.makedirs('models', exist_ok=True)
        
        for name, model in trained_models.items():
            model_path = f'models/{name}_model.joblib'
            joblib.dump(model, model_path)
            logger.info(f"Saved {name} to {model_path}")
        
        # Save preprocessor
        preprocessor_path = 'models/preprocessor.joblib'
        preprocessor.save(preprocessor_path)
        logger.info(f"Saved preprocessor to {preprocessor_path}")
        
        # Save results summary
        results_path = 'models/training_results.joblib'
        joblib.dump(results, results_path)
        
        logger.info("Training completed successfully!")
        logger.info("Results summary:")
        for name, metrics in results.items():
            logger.info(f"  {name}: Accuracy={metrics['accuracy']:.3f}, ROC-AUC={metrics['roc_auc']:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        return False

if __name__ == "__main__":
    success = train_simple_models()
    if success:
        print("✅ Models trained successfully!")
    else:
        print("❌ Training failed!")