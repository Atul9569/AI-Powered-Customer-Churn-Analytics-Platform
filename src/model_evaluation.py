import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import logging
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluation for customer churn prediction
    """
    
    def __init__(self):
        self.evaluation_results = {}
        
    def evaluate_model(self, 
                      model: Any, 
                      X_test: np.ndarray, 
                      y_test: np.ndarray,
                      model_name: str) -> Dict[str, float]:
        """
        Evaluate a single model with comprehensive metrics
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Store detailed results
        self.evaluation_results[model_name] = {
            'metrics': metrics,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, "
                   f"Precision: {metrics['precision']:.4f}, "
                   f"Recall: {metrics['recall']:.4f}, "
                   f"F1: {metrics['f1_score']:.4f}, "
                   f"ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def evaluate_all_models(self, 
                           models: Dict[str, Any], 
                           X_test: np.ndarray, 
                           y_test: np.ndarray) -> pd.DataFrame:
        """
        Evaluate multiple models
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test target
            
        Returns:
            pd.DataFrame: Comparison of model performance
        """
        logger.info("Evaluating all models...")
        
        results = []
        
        for model_name, model in models.items():
            try:
                metrics = self.evaluate_model(model, X_test, y_test, model_name)
                metrics['model'] = model_name
                results.append(metrics)
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
                continue
        
        results_df = pd.DataFrame(results)
        return results_df
    
    def plot_confusion_matrices(self, save_path: str = None) -> None:
        """
        Plot confusion matrices for all models
        
        Args:
            save_path: Path to save the plot
        """
        n_models = len(self.evaluation_results)
        if n_models == 0:
            logger.warning("No evaluation results found")
            return
        
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, results) in enumerate(self.evaluation_results.items()):
            cm = results['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['No Churn', 'Churn'],
                       yticklabels=['No Churn', 'Churn'],
                       ax=axes[idx])
            
            axes[idx].set_title(f'{model_name.replace("_", " ").title()}\nConfusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrices saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self, save_path: str = None) -> None:
        """
        Plot ROC curves for all models
        
        Args:
            save_path: Path to save the plot
        """
        if not self.evaluation_results:
            logger.warning("No evaluation results found")
            return
        
        plt.figure(figsize=(10, 8))
        
        for model_name, results in self.evaluation_results.items():
            y_true = results['y_true']
            y_pred_proba = results['y_pred_proba']
            
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            auc_score = roc_auc_score(y_true, y_pred_proba)
            
            plt.plot(fpr, tpr, label=f'{model_name.replace("_", " ").title()} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curves(self, save_path: str = None) -> None:
        """
        Plot Precision-Recall curves for all models
        
        Args:
            save_path: Path to save the plot
        """
        if not self.evaluation_results:
            logger.warning("No evaluation results found")
            return
        
        plt.figure(figsize=(10, 8))
        
        for model_name, results in self.evaluation_results.items():
            y_true = results['y_true']
            y_pred_proba = results['y_pred_proba']
            
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            
            plt.plot(recall, precision, label=f'{model_name.replace("_", " ").title()}')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-Recall curves saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, 
                               models: Dict[str, Any], 
                               feature_names: List[str],
                               save_path: str = None) -> None:
        """
        Plot feature importance for tree-based models
        
        Args:
            models: Dictionary of trained models
            feature_names: List of feature names
            save_path: Path to save the plot
        """
        # Filter models that have feature_importances_
        importance_models = {}
        for name, model in models.items():
            if hasattr(model, 'feature_importances_'):
                importance_models[name] = model
        
        if not importance_models:
            logger.warning("No models with feature importance found")
            return
        
        n_models = len(importance_models)
        fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 6))
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, model) in enumerate(importance_models.items()):
            importance = model.feature_importances_
            
            # Get top 15 features
            indices = np.argsort(importance)[-15:]
            
            axes[idx].barh(range(len(indices)), importance[indices])
            axes[idx].set_yticks(range(len(indices)))
            axes[idx].set_yticklabels([feature_names[i] for i in indices])
            axes[idx].set_xlabel('Feature Importance')
            axes[idx].set_title(f'{model_name.replace("_", " ").title()}\nTop 15 Features')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plots saved to {save_path}")
        
        plt.show()
    
    def generate_model_comparison_report(self) -> pd.DataFrame:
        """
        Generate a comprehensive model comparison report
        
        Returns:
            pd.DataFrame: Model comparison report
        """
        if not self.evaluation_results:
            logger.warning("No evaluation results found")
            return pd.DataFrame()
        
        report_data = []
        
        for model_name, results in self.evaluation_results.items():
            metrics = results['metrics']
            
            report_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}"
            })
        
        report_df = pd.DataFrame(report_data)
        return report_df
    
    def save_evaluation_results(self, filepath: str) -> None:
        """
        Save evaluation results to file
        
        Args:
            filepath: Path to save the results
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        
        for model_name, results in self.evaluation_results.items():
            serializable_results[model_name] = {
                'metrics': results['metrics'],
                'confusion_matrix': results['confusion_matrix'].tolist(),
                'classification_report': results['classification_report']
            }
        
        joblib.dump(serializable_results, filepath)
        logger.info(f"Evaluation results saved to {filepath}")

def main():
    """
    Main evaluation pipeline
    """
    logger.info("Starting model evaluation pipeline...")
    
    try:
        # Load data and models
        from data_loader import TelcoDataLoader
        from preprocessing import prepare_data
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
        
        # Load trained models
        trainer = ChurnModelTrainer()
        models = trainer.load_models('models')
        
        if not models:
            logger.error("No trained models found. Please run training first.")
            return
        
        # Initialize evaluator
        evaluator = ModelEvaluator()
        
        # Evaluate all models
        results_df = evaluator.evaluate_all_models(models, X_test, y_test)
        
        print("\nModel Comparison Results:")
        print("=" * 80)
        print(results_df.to_string(index=False))
        
        # Generate visualizations
        os.makedirs('plots', exist_ok=True)
        
        evaluator.plot_confusion_matrices('plots/confusion_matrices.png')
        evaluator.plot_roc_curves('plots/roc_curves.png')
        evaluator.plot_precision_recall_curves('plots/precision_recall_curves.png')
        
        # Feature importance for tree-based models
        feature_names = preprocessor.get_feature_names()
        evaluator.plot_feature_importance(models, feature_names, 'plots/feature_importance.png')
        
        # Save evaluation results
        evaluator.save_evaluation_results('models/evaluation_results.joblib')
        
        # Generate and save report
        report = evaluator.generate_model_comparison_report()
        report.to_csv('models/model_comparison_report.csv', index=False)
        
        logger.info("Model evaluation pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in evaluation pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
