"""
Model Development and Evaluation for Customer Churn Prediction
This notebook demonstrates the complete machine learning pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('default')
sns.set_palette("husl")

def run_complete_pipeline():
    """Run the complete ML pipeline"""
    print("=" * 80)
    print("CUSTOMER CHURN PREDICTION - MODEL DEVELOPMENT PIPELINE")
    print("=" * 80)
    
    # Step 1: Load and prepare data
    print("\n🔄 Step 1: Loading and preparing data...")
    df = load_and_prepare_data()
    if df is None:
        return
    
    # Step 2: Data preprocessing
    print("\n🔧 Step 2: Data preprocessing...")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    
    # Step 3: Model training
    print("\n🤖 Step 3: Training models...")
    models = train_models(X_train, y_train)
    
    # Step 4: Model evaluation
    print("\n📊 Step 4: Evaluating models...")
    evaluation_results = evaluate_models(models, X_test, y_test, preprocessor)
    
    # Step 5: Model comparison and selection
    print("\n🏆 Step 5: Model comparison and selection...")
    best_model_name = compare_and_select_models(evaluation_results)
    
    # Step 6: Feature importance analysis
    print("\n🔍 Step 6: Feature importance analysis...")
    analyze_feature_importance(models, preprocessor, best_model_name)
    
    # Step 7: Business insights
    print("\n💼 Step 7: Generating business insights...")
    generate_business_insights(evaluation_results, df)
    
    print("\n✅ Model development pipeline completed successfully!")

def load_and_prepare_data():
    """Load and prepare the dataset"""
    try:
        from src.data_loader import TelcoDataLoader
        
        loader = TelcoDataLoader()
        df = loader.load_data()
        
        if df is None:
            print("❌ Error: Could not load data")
            return None
        
        print(f"✅ Data loaded successfully: {df.shape}")
        print(f"   • Features: {df.shape[1] - 1}")
        print(f"   • Samples: {df.shape[0]}")
        print(f"   • Churn rate: {(df['Churn'] == 'Yes').mean():.1%}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None

def preprocess_data(df):
    """Preprocess the data"""
    try:
        from src.data_loader import TelcoDataLoader
        from src.preprocessing import prepare_data
        
        loader = TelcoDataLoader()
        X, y = loader.split_features_target(df)
        
        # Prepare data with train-test split and preprocessing
        X_train, X_test, y_train, y_test, preprocessor = prepare_data(X, y, test_size=0.2, random_state=42)
        
        print(f"✅ Data preprocessing completed:")
        print(f"   • Training samples: {X_train.shape[0]}")
        print(f"   • Test samples: {X_test.shape[0]}")
        print(f"   • Features after preprocessing: {X_train.shape[1]}")
        print(f"   • Training churn rate: {y_train.mean():.1%}")
        print(f"   • Test churn rate: {y_test.mean():.1%}")
        
        return X_train, X_test, y_train, y_test, preprocessor
        
    except Exception as e:
        print(f"❌ Error in preprocessing: {str(e)}")
        raise

def train_models(X_train, y_train):
    """Train multiple models"""
    try:
        from src.model_training import ChurnModelTrainer
        
        # Initialize trainer
        trainer = ChurnModelTrainer(random_state=42)
        
        print("🔧 Training models with hyperparameter optimization...")
        
        # Train all models
        models = trainer.train_all_models(X_train, y_train, optimize=True)
        
        print(f"✅ Successfully trained {len(models)} models:")
        for model_name in models.keys():
            print(f"   • {model_name.replace('_', ' ').title()}")
        
        # Save models
        trainer.save_models('models')
        print("💾 Models saved to 'models' directory")
        
        return models
        
    except Exception as e:
        print(f"❌ Error in model training: {str(e)}")
        raise

def evaluate_models(models, X_test, y_test, preprocessor):
    """Evaluate all trained models"""
    try:
        from src.model_evaluation import ModelEvaluator
        
        # Initialize evaluator
        evaluator = ModelEvaluator()
        
        # Evaluate all models
        results_df = evaluator.evaluate_all_models(models, X_test, y_test)
        
        print("📊 Model Evaluation Results:")
        print("=" * 60)
        for _, row in results_df.iterrows():
            print(f"{row['model'].replace('_', ' ').title():>20}: "
                  f"Accuracy={row['accuracy']:.3f}, "
                  f"F1={row['f1_score']:.3f}, "
                  f"ROC-AUC={row['roc_auc']:.3f}")
        
        # Generate visualizations
        print("\n📈 Generating evaluation plots...")
        
        # Confusion matrices
        evaluator.plot_confusion_matrices()
        
        # ROC curves
        evaluator.plot_roc_curves()
        
        # Precision-Recall curves
        evaluator.plot_precision_recall_curves()
        
        # Feature importance (for tree-based models)
        feature_names = preprocessor.get_feature_names()
        evaluator.plot_feature_importance(models, feature_names)
        
        return evaluator.evaluation_results
        
    except Exception as e:
        print(f"❌ Error in model evaluation: {str(e)}")
        raise

def compare_and_select_models(evaluation_results):
    """Compare models and select the best one"""
    print("\n🏆 Model Selection Analysis:")
    print("=" * 50)
    
    # Create comparison dataframe
    comparison_data = []
    for model_name, results in evaluation_results.items():
        metrics = results['metrics']
        comparison_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'ROC-AUC': metrics['roc_auc']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Find best model based on F1-score (balanced metric for churn prediction)
    best_idx = comparison_df['F1-Score'].idxmax()
    best_model = comparison_df.loc[best_idx]
    
    print(f"🥇 Best Model: {best_model['Model']}")
    print(f"   • Accuracy: {best_model['Accuracy']:.3f}")
    print(f"   • Precision: {best_model['Precision']:.3f}")
    print(f"   • Recall: {best_model['Recall']:.3f}")
    print(f"   • F1-Score: {best_model['F1-Score']:.3f}")
    print(f"   • ROC-AUC: {best_model['ROC-AUC']:.3f}")
    
    # Check if model meets success criteria
    target_accuracy = 0.89
    if best_model['Accuracy'] >= target_accuracy:
        print(f"✅ Target accuracy ({target_accuracy:.1%}) achieved!")
    else:
        print(f"⚠️ Target accuracy ({target_accuracy:.1%}) not yet achieved. Current: {best_model['Accuracy']:.1%}")
    
    # Display full comparison table
    print(f"\n📋 Complete Model Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Visualize model comparison
    visualize_model_comparison(comparison_df)
    
    return comparison_df.loc[best_idx, 'Model'].lower().replace(' ', '_')

def visualize_model_comparison(comparison_df):
    """Create visualizations for model comparison"""
    
    # Prepare data for plotting
    models = comparison_df['Model']
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        bars = axes[i].bar(models, comparison_df[metric], alpha=0.7)
        axes[i].set_title(f'{metric} Comparison')
        axes[i].set_ylabel(metric)
        axes[i].set_ylim(0, 1)
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, comparison_df[metric]):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    # Radar chart for overall comparison
    axes[5].remove()
    ax_radar = fig.add_subplot(2, 3, 6, projection='polar')
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for idx, model in enumerate(models):
        values = comparison_df.iloc[idx][metrics].tolist()
        values += values[:1]  # Complete the circle
        
        ax_radar.plot(angles, values, 'o-', linewidth=2, label=model)
        ax_radar.fill(angles, values, alpha=0.25)
    
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(metrics)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title('Overall Model Comparison (Radar Chart)')
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    plt.tight_layout()
    plt.show()

def analyze_feature_importance(models, preprocessor, best_model_name):
    """Analyze feature importance for the best model"""
    print(f"\n🔍 Feature Importance Analysis for {best_model_name.replace('_', ' ').title()}:")
    print("=" * 60)
    
    try:
        feature_names = preprocessor.get_feature_names()
        
        # Get the best model
        best_model = models.get(best_model_name)
        if best_model is None:
            print(f"❌ Model {best_model_name} not found")
            return
        
        # Check if model has feature importance
        if hasattr(best_model, 'feature_importances_'):
            importance = best_model.feature_importances_
            
            # Create feature importance dataframe
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            print(f"📊 Top 15 Most Important Features:")
            for idx, row in feature_importance_df.head(15).iterrows():
                print(f"   {row['Feature']:<25}: {row['Importance']:.4f}")
            
            # Visualize top features
            plt.figure(figsize=(12, 8))
            top_features = feature_importance_df.head(15)
            bars = plt.barh(range(len(top_features)), top_features['Importance'])
            plt.yticks(range(len(top_features)), top_features['Feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top 15 Feature Importance - {best_model_name.replace("_", " ").title()}')
            plt.gca().invert_yaxis()
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, top_features['Importance'])):
                plt.text(value + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{value:.3f}', va='center')
            
            plt.tight_layout()
            plt.show()
            
        elif hasattr(best_model, 'coef_'):
            # For logistic regression
            importance = np.abs(best_model.coef_[0])
            
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            print(f"📊 Top 15 Most Important Features (Logistic Regression Coefficients):")
            for idx, row in feature_importance_df.head(15).iterrows():
                print(f"   {row['Feature']:<25}: {row['Importance']:.4f}")
        else:
            print(f"⚠️ Feature importance not available for {best_model_name}")
            
    except Exception as e:
        print(f"❌ Error in feature importance analysis: {str(e)}")

def generate_business_insights(evaluation_results, df):
    """Generate business insights from the model results"""
    print("\n💼 Business Insights and Recommendations:")
    print("=" * 60)
    
    # Get the best model results
    best_f1 = 0
    best_model_name = ""
    best_results = None
    
    for model_name, results in evaluation_results.items():
        if results['metrics']['f1_score'] > best_f1:
            best_f1 = results['metrics']['f1_score']
            best_model_name = model_name
            best_results = results
    
    if best_results:
        metrics = best_results['metrics']
        cm = best_results['confusion_matrix']
        
        # Calculate business metrics
        total_customers = len(df)
        actual_churned = (df['Churn'] == 'Yes').sum()
        churn_rate = actual_churned / total_customers
        
        # Calculate prediction accuracy metrics
        true_negatives, false_positives, false_negatives, true_positives = cm.ravel()
        
        # Revenue calculations (example values)
        avg_monthly_revenue = df['MonthlyCharges'].mean()
        avg_customer_lifetime = df['tenure'].mean()
        
        print(f"🎯 Model Performance Insights:")
        print(f"   • Best Model: {best_model_name.replace('_', ' ').title()}")
        print(f"   • Accuracy: {metrics['accuracy']:.1%}")
        print(f"   • Precision: {metrics['precision']:.1%} (of predicted churners, {metrics['precision']:.1%} actually churn)")
        print(f"   • Recall: {metrics['recall']:.1%} (model catches {metrics['recall']:.1%} of actual churners)")
        
        print(f"\n💰 Business Impact Analysis:")
        print(f"   • Total Customers: {total_customers:,}")
        print(f"   • Current Churn Rate: {churn_rate:.1%}")
        print(f"   • Average Monthly Revenue per Customer: ${avg_monthly_revenue:.2f}")
        
        # Calculate potential savings
        customers_correctly_identified = true_positives
        revenue_at_risk_identified = customers_correctly_identified * avg_monthly_revenue * 12  # Annual
        
        print(f"   • High-risk customers correctly identified: {customers_correctly_identified}")
        print(f"   • Annual revenue at risk (identified): ${revenue_at_risk_identified:,.2f}")
        
        print(f"\n📈 Actionable Recommendations:")
        
        # Precision-based recommendations
        if metrics['precision'] > 0.8:
            print("   ✅ High Precision: Focus on intensive retention for predicted churners")
        else:
            print("   ⚠️ Moderate Precision: Use tiered intervention approach")
        
        # Recall-based recommendations  
        if metrics['recall'] > 0.8:
            print("   ✅ High Recall: Model catches most at-risk customers")
        else:
            print("   ⚠️ Moderate Recall: Consider ensemble methods or feature engineering")
        
        # Specific business actions
        print(f"\n🎯 Recommended Actions:")
        print("   1. Immediate Actions:")
        print("      • Deploy model for real-time churn scoring")
        print("      • Create automated alerts for high-risk customers")
        print("      • Design retention campaigns for predicted churners")
        
        print("   2. Medium-term Strategies:")
        print("      • Implement A/B testing for retention interventions")
        print("      • Develop customer journey optimization")
        print("      • Create predictive customer success programs")
        
        print("   3. Long-term Improvements:")
        print("      • Collect additional behavioral data")
        print("      • Implement real-time feature updates")
        print("      • Develop segment-specific retention strategies")
        
        # ROI estimation
        if metrics['precision'] > 0.5 and metrics['recall'] > 0.5:
            # Assume retention intervention cost and success rate
            intervention_cost_per_customer = 50  # Example cost
            retention_success_rate = 0.3  # 30% of interventions successful
            
            customers_to_target = true_positives + false_positives
            intervention_cost = customers_to_target * intervention_cost_per_customer
            customers_retained = true_positives * retention_success_rate
            revenue_saved = customers_retained * avg_monthly_revenue * 12
            
            roi = ((revenue_saved - intervention_cost) / intervention_cost) * 100
            
            print(f"\n📊 Estimated ROI Analysis:")
            print(f"   • Customers to target: {customers_to_target}")
            print(f"   • Intervention cost: ${intervention_cost:,.2f}")
            print(f"   • Estimated customers retained: {customers_retained:.0f}")
            print(f"   • Revenue saved: ${revenue_saved:,.2f}")
            print(f"   • Estimated ROI: {roi:.0f}%")

def main():
    """Main function to run the complete pipeline"""
    run_complete_pipeline()

if __name__ == "__main__":
    main()
