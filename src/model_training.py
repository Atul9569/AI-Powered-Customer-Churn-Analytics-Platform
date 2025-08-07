import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import logging
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChurnModelTrainer:
    """
    Model trainer for customer churn prediction
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_params = {}
        self.cv_results = {}
        
    def initialize_models(self) -> Dict[str, Any]:
        """
        Initialize machine learning models
        
        Returns:
            Dict[str, Any]: Dictionary of models
        """
        models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ),
            'random_forest': RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=-1
            ),
            'xgboost': XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                verbosity=0
            )
        }
        
        return models
    
    def get_param_grids(self) -> Dict[str, Dict]:
        """
        Get parameter grids for hyperparameter tuning
        
        Returns:
            Dict[str, Dict]: Parameter grids for each model
        """
        param_grids = {
            'logistic_regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        }
        
        return param_grids
    
    def train_model(self, 
                   model_name: str, 
                   X_train: np.ndarray, 
                   y_train: np.ndarray,
                   optimize: bool = True,
                   cv_folds: int = 5) -> Any:
        """
        Train a single model with optional hyperparameter optimization
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training target
            optimize: Whether to perform hyperparameter optimization
            cv_folds: Number of cross-validation folds
            
        Returns:
            Trained model
        """
        logger.info(f"Training {model_name}...")
        
        models = self.initialize_models()
        param_grids = self.get_param_grids()
        
        if model_name not in models:
            raise ValueError(f"Model {model_name} not supported")
        
        model = models[model_name]
        
        if optimize and model_name in param_grids:
            logger.info(f"Optimizing hyperparameters for {model_name}...")
            
            # Use RandomizedSearchCV for faster optimization
            if model_name == 'xgboost':
                search = RandomizedSearchCV(
                    model, 
                    param_grids[model_name],
                    n_iter=20,
                    cv=cv_folds,
                    scoring='accuracy',
                    n_jobs=-1,
                    random_state=self.random_state,
                    verbose=0
                )
            else:
                search = GridSearchCV(
                    model,
                    param_grids[model_name],
                    cv=cv_folds,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=0
                )
            
            search.fit(X_train, y_train)
            
            self.best_params[model_name] = search.best_params_
            self.cv_results[model_name] = {
                'best_score': search.best_score_,
                'best_params': search.best_params_
            }
            
            logger.info(f"Best parameters for {model_name}: {search.best_params_}")
            logger.info(f"Best CV score for {model_name}: {search.best_score_:.4f}")
            
            model = search.best_estimator_
        else:
            model.fit(X_train, y_train)
        
        self.models[model_name] = model
        return model
    
    def train_all_models(self, 
                        X_train: np.ndarray, 
                        y_train: np.ndarray,
                        optimize: bool = True) -> Dict[str, Any]:
        """
        Train all models
        
        Args:
            X_train: Training features
            y_train: Training target
            optimize: Whether to perform hyperparameter optimization
            
        Returns:
            Dict[str, Any]: Dictionary of trained models
        """
        logger.info("Training all models...")
        
        models = self.initialize_models()
        
        for model_name in models.keys():
            try:
                self.train_model(model_name, X_train, y_train, optimize)
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        logger.info(f"Successfully trained {len(self.models)} models")
        return self.models
    
    def save_models(self, models_dir: str = 'models') -> None:
        """
        Save trained models
        
        Args:
            models_dir: Directory to save models
        """
        os.makedirs(models_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = os.path.join(models_dir, f'{model_name}_model.joblib')
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_name} to {model_path}")
        
        # Save training results
        results_path = os.path.join(models_dir, 'training_results.joblib')
        joblib.dump({
            'best_params': self.best_params,
            'cv_results': self.cv_results
        }, results_path)
        logger.info(f"Saved training results to {results_path}")
    
    def load_models(self, models_dir: str = 'models') -> Dict[str, Any]:
        """
        Load trained models
        
        Args:
            models_dir: Directory containing saved models
            
        Returns:
            Dict[str, Any]: Dictionary of loaded models
        """
        models = {}
        
        for model_name in ['logistic_regression', 'random_forest', 'xgboost']:
            model_path = os.path.join(models_dir, f'{model_name}_model.joblib')
            if os.path.exists(model_path):
                models[model_name] = joblib.load(model_path)
                logger.info(f"Loaded {model_name} from {model_path}")
        
        self.models = models
        return models
    
    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with a specific model
        
        Args:
            model_name: Name of the model
            X: Features for prediction
            
        Returns:
            np.ndarray: Predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        return self.models[model_name].predict(X)
    
    def predict_proba(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities with a specific model
        
        Args:
            model_name: Name of the model
            X: Features for prediction
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        return self.models[model_name].predict_proba(X)

def main():
    """
    Main training pipeline
    """
    logger.info("Starting model training pipeline...")
    
    try:
        # Import data loading and preprocessing
        from data_loader import TelcoDataLoader
        from preprocessing import prepare_data
        
        # Load data
        loader = TelcoDataLoader()
        data = loader.load_data()
        
        if data is None:
            logger.error("Failed to load data")
            return
        
        # Split features and target
        X, y = loader.split_features_target(data)
        
        # Prepare data
        X_train, X_test, y_train, y_test, preprocessor = prepare_data(X, y)
        
        # Initialize trainer
        trainer = ChurnModelTrainer()
        
        # Train all models
        models = trainer.train_all_models(X_train, y_train, optimize=True)
        
        # Save models and preprocessor
        os.makedirs('models', exist_ok=True)
        trainer.save_models('models')
        preprocessor.save('models/preprocessor.joblib')
        
        # Quick evaluation
        logger.info("Quick evaluation on test set:")
        for model_name, model in models.items():
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            logger.info(f"{model_name}: Accuracy = {accuracy:.4f}")
        
        logger.info("Model training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
