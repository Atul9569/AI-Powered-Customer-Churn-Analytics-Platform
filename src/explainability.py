import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import logging
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")

class ShapExplainer:
    """
    SHAP-based model explainability for customer churn prediction
    """
    
    def __init__(self):
        self.explainers = {}
        self.shap_values = {}
        self.feature_names = None
        
    def create_explainer(self, 
                        model: Any, 
                        X_background: np.ndarray,
                        model_name: str,
                        feature_names: List[str]) -> None:
        """
        Create SHAP explainer for a model
        
        Args:
            model: Trained model
            X_background: Background dataset for SHAP
            model_name: Name of the model
            feature_names: List of feature names
        """
        if not SHAP_AVAILABLE:
            logger.error("SHAP is not available. Cannot create explainer.")
            return
        
        logger.info(f"Creating SHAP explainer for {model_name}...")
        
        self.feature_names = feature_names
        
        try:
            # Choose explainer type based on model
            if hasattr(model, 'predict_proba'):
                if 'random_forest' in model_name.lower() or 'xgboost' in model_name.lower():
                    # Tree explainer for tree-based models
                    explainer = shap.TreeExplainer(model)
                else:
                    # Use a subset of background data for efficiency
                    background_sample = shap.sample(X_background, min(100, len(X_background)))
                    explainer = shap.Explainer(model.predict_proba, background_sample)
                
                self.explainers[model_name] = explainer
                logger.info(f"SHAP explainer created for {model_name}")
                
            else:
                logger.warning(f"Model {model_name} doesn't support probability prediction")
                
        except Exception as e:
            logger.error(f"Error creating SHAP explainer for {model_name}: {str(e)}")
    
    def calculate_shap_values(self, 
                             model_name: str, 
                             X_explain: np.ndarray,
                             max_samples: int = 100) -> Optional[np.ndarray]:
        """
        Calculate SHAP values for given samples
        
        Args:
            model_name: Name of the model
            X_explain: Samples to explain
            max_samples: Maximum number of samples to explain
            
        Returns:
            Optional[np.ndarray]: SHAP values or None if error
        """
        if not SHAP_AVAILABLE:
            logger.error("SHAP is not available.")
            return None
        
        if model_name not in self.explainers:
            logger.error(f"No explainer found for {model_name}")
            return None
        
        try:
            # Limit samples for computational efficiency
            X_sample = X_explain[:max_samples] if len(X_explain) > max_samples else X_explain
            
            logger.info(f"Calculating SHAP values for {model_name} on {len(X_sample)} samples...")
            
            explainer = self.explainers[model_name]
            
            # Calculate SHAP values
            if 'random_forest' in model_name.lower() or 'xgboost' in model_name.lower():
                shap_values = explainer.shap_values(X_sample)
                # For binary classification, take the positive class
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            else:
                shap_values = explainer(X_sample)
                if hasattr(shap_values, 'values'):
                    shap_values = shap_values.values
                    if len(shap_values.shape) == 3:  # Multi-output
                        shap_values = shap_values[:, :, 1]  # Take positive class
            
            self.shap_values[model_name] = shap_values
            logger.info(f"SHAP values calculated for {model_name}")
            
            return shap_values
            
        except Exception as e:
            logger.error(f"Error calculating SHAP values for {model_name}: {str(e)}")
            return None
    
    def plot_feature_importance(self, 
                               model_name: str,
                               save_path: str = None) -> None:
        """
        Plot SHAP feature importance
        
        Args:
            model_name: Name of the model
            save_path: Path to save the plot
        """
        if not SHAP_AVAILABLE:
            logger.error("SHAP is not available.")
            return
        
        if model_name not in self.shap_values:
            logger.error(f"No SHAP values found for {model_name}")
            return
        
        try:
            shap_values = self.shap_values[model_name]
            
            plt.figure(figsize=(10, 8))
            
            # Calculate mean absolute SHAP values for feature importance
            feature_importance = np.mean(np.abs(shap_values), axis=0)
            
            # Get top 15 features
            top_indices = np.argsort(feature_importance)[-15:]
            
            plt.barh(range(len(top_indices)), feature_importance[top_indices])
            
            if self.feature_names:
                plt.yticks(range(len(top_indices)), [self.feature_names[i] for i in top_indices])
            else:
                plt.yticks(range(len(top_indices)), [f'Feature {i}' for i in top_indices])
            
            plt.xlabel('Mean |SHAP Value|')
            plt.title(f'SHAP Feature Importance - {model_name.replace("_", " ").title()}')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"SHAP feature importance plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting SHAP feature importance: {str(e)}")
    
    def plot_summary_plot(self, 
                         model_name: str,
                         X_data: np.ndarray,
                         save_path: str = None) -> None:
        """
        Plot SHAP summary plot
        
        Args:
            model_name: Name of the model
            X_data: Feature data for the explained samples
            save_path: Path to save the plot
        """
        if not SHAP_AVAILABLE:
            logger.error("SHAP is not available.")
            return
        
        if model_name not in self.shap_values:
            logger.error(f"No SHAP values found for {model_name}")
            return
        
        try:
            shap_values = self.shap_values[model_name]
            
            plt.figure(figsize=(10, 8))
            
            # Create feature names for the plot
            feature_names_plot = self.feature_names if self.feature_names else [f'Feature {i}' for i in range(X_data.shape[1])]
            
            shap.summary_plot(
                shap_values, 
                X_data, 
                feature_names=feature_names_plot,
                show=False,
                max_display=15
            )
            
            plt.title(f'SHAP Summary Plot - {model_name.replace("_", " ").title()}')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"SHAP summary plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting SHAP summary plot: {str(e)}")
    
    def explain_single_prediction(self, 
                                 model_name: str,
                                 X_single: np.ndarray,
                                 feature_values: np.ndarray = None,
                                 save_path: str = None) -> None:
        """
        Explain a single prediction with SHAP
        
        Args:
            model_name: Name of the model
            X_single: Single sample to explain (1D array)
            feature_values: Original feature values for display
            save_path: Path to save the plot
        """
        if not SHAP_AVAILABLE:
            logger.error("SHAP is not available.")
            return
        
        if model_name not in self.explainers:
            logger.error(f"No explainer found for {model_name}")
            return
        
        try:
            explainer = self.explainers[model_name]
            
            # Ensure X_single is 2D
            if X_single.ndim == 1:
                X_single = X_single.reshape(1, -1)
            
            # Calculate SHAP values for single prediction
            if 'random_forest' in model_name.lower() or 'xgboost' in model_name.lower():
                shap_values = explainer.shap_values(X_single)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1][0]  # Take positive class, first sample
                else:
                    shap_values = shap_values[0]
            else:
                shap_values = explainer(X_single)
                if hasattr(shap_values, 'values'):
                    shap_values = shap_values.values[0]
                    if len(shap_values.shape) == 2:  # Multi-output
                        shap_values = shap_values[:, 1]  # Take positive class
            
            # Use provided feature values or the scaled values
            display_values = feature_values if feature_values is not None else X_single[0]
            
            plt.figure(figsize=(10, 8))
            
            # Create waterfall plot manually if SHAP waterfall is not available
            feature_names_plot = self.feature_names if self.feature_names else [f'Feature {i}' for i in range(len(shap_values))]
            
            # Get top contributing features
            top_indices = np.argsort(np.abs(shap_values))[-10:]
            
            colors = ['red' if val < 0 else 'blue' for val in shap_values[top_indices]]
            
            plt.barh(range(len(top_indices)), shap_values[top_indices], color=colors)
            plt.yticks(range(len(top_indices)), [feature_names_plot[i] for i in top_indices])
            plt.xlabel('SHAP Value')
            plt.title(f'SHAP Explanation - {model_name.replace("_", " ").title()}')
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # Add value annotations
            for i, (idx, val) in enumerate(zip(top_indices, shap_values[top_indices])):
                plt.text(val + (0.01 if val >= 0 else -0.01), i, f'{val:.3f}', 
                        ha='left' if val >= 0 else 'right', va='center')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"SHAP single prediction plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error explaining single prediction: {str(e)}")
    
    def get_top_features_for_prediction(self, 
                                       model_name: str,
                                       X_single: np.ndarray,
                                       top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get top contributing features for a single prediction
        
        Args:
            model_name: Name of the model
            X_single: Single sample to explain
            top_k: Number of top features to return
            
        Returns:
            List[Dict[str, Any]]: Top contributing features with their SHAP values
        """
        if not SHAP_AVAILABLE:
            logger.error("SHAP is not available.")
            return []
        
        if model_name not in self.explainers:
            logger.error(f"No explainer found for {model_name}")
            return []
        
        try:
            explainer = self.explainers[model_name]
            
            # Ensure X_single is 2D
            if X_single.ndim == 1:
                X_single = X_single.reshape(1, -1)
            
            # Calculate SHAP values
            if 'random_forest' in model_name.lower() or 'xgboost' in model_name.lower():
                shap_values = explainer.shap_values(X_single)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1][0]
                else:
                    shap_values = shap_values[0]
            else:
                shap_values = explainer(X_single)
                if hasattr(shap_values, 'values'):
                    shap_values = shap_values.values[0]
                    if len(shap_values.shape) == 2:
                        shap_values = shap_values[:, 1]
            
            # Get top contributing features
            top_indices = np.argsort(np.abs(shap_values))[-top_k:][::-1]
            
            feature_names_list = self.feature_names if self.feature_names else [f'Feature {i}' for i in range(len(shap_values))]
            
            top_features = []
            for idx in top_indices:
                top_features.append({
                    'feature_name': feature_names_list[idx],
                    'shap_value': float(shap_values[idx]),
                    'feature_value': float(X_single[0][idx]),
                    'contribution': 'Increases churn risk' if shap_values[idx] > 0 else 'Decreases churn risk'
                })
            
            return top_features
            
        except Exception as e:
            logger.error(f"Error getting top features: {str(e)}")
            return []

def main():
    """
    Main explainability pipeline
    """
    if not SHAP_AVAILABLE:
        logger.error("SHAP is required for explainability analysis. Install with: pip install shap")
        return
    
    logger.info("Starting explainability analysis...")
    
    try:
        # Load data and models
        from data_loader import TelcoDataLoader
        from preprocessing import prepare_data, ChurnPreprocessor
        from model_training import ChurnModelTrainer
        
        # Load data
        loader = TelcoDataLoader()
        data = loader.load_data()
        
        if data is None:
            logger.error("Failed to load data")
            return
        
        # Prepare data
        X, y = loader.split_features_target(data)
        X_train, X_test, y_train, y_test, preprocessor = prepare_data(X, y)
        
        # Load models
        trainer = ChurnModelTrainer()
        models = trainer.load_models('models')
        
        if not models:
            logger.error("No trained models found")
            return
        
        # Initialize explainer
        explainer = ShapExplainer()
        
        # Create explainers for each model
        feature_names = preprocessor.get_feature_names()
        
        for model_name, model in models.items():
            explainer.create_explainer(model, X_train, model_name, feature_names)
        
        # Calculate SHAP values
        for model_name in models.keys():
            explainer.calculate_shap_values(model_name, X_test, max_samples=50)
        
        # Generate plots
        import os
        os.makedirs('plots/shap', exist_ok=True)
        
        for model_name in models.keys():
            if model_name in explainer.shap_values:
                explainer.plot_feature_importance(
                    model_name, 
                    f'plots/shap/{model_name}_feature_importance.png'
                )
                
                explainer.plot_summary_plot(
                    model_name,
                    X_test[:50],
                    f'plots/shap/{model_name}_summary_plot.png'
                )
        
        # Example single prediction explanation
        if len(X_test) > 0:
            sample_idx = 0
            for model_name in models.keys():
                if model_name in explainer.explainers:
                    explainer.explain_single_prediction(
                        model_name,
                        X_test[sample_idx],
                        save_path=f'plots/shap/{model_name}_single_prediction.png'
                    )
                    break
        
        logger.info("Explainability analysis completed!")
        
    except Exception as e:
        logger.error(f"Error in explainability pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
