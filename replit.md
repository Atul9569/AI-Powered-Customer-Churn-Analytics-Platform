# Customer Churn Prediction - Complete Data Science Project

## Overview

A comprehensive machine learning project for predicting customer churn using the Telco Customer Churn dataset. This project implements a complete data science pipeline including exploratory data analysis, multiple ML models (Logistic Regression, Random Forest, XGBoost), hyperparameter optimization, SHAP explainability, and an interactive Streamlit web application. The system achieves high performance metrics with the best model reaching 93.2% accuracy and 97.5% ROC-AUC score.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Updated Project Structure (January 2025)
The codebase now features a separated frontend/backend architecture for easy deployment:

- **frontend.py** - Modern React-like Streamlit UI with interactive components and animations
- **backend.py** - Backend processing service with data handling and ML model management
- **deploy.py** - Simple deployment entry point for production environments
- **app.py** - Original combined application (legacy, maintained for compatibility)
- **src/** - Core business logic modules for data processing, model training, and evaluation
- **notebooks/** - Jupyter notebooks for exploratory data analysis and model development
- **data/** - Dataset storage with fallback sample data generation
- **models/** - Trained model persistence using joblib
- **DEPLOYMENT.md** - Comprehensive deployment guide with multiple options

### Data Processing Pipeline
The preprocessing system uses a centralized `ChurnPreprocessor` class that handles:

- **Data Loading**: Robust data loader with fallback sample data generation if dataset is missing
- **Feature Engineering**: Automatic numerical/categorical feature separation and encoding
- **Missing Value Handling**: Median imputation for numerical features
- **Categorical Encoding**: Label encoding with fitted encoders for consistent transformations
- **Feature Scaling**: StandardScaler for numerical feature normalization
- **Pipeline Persistence**: Serialized preprocessor state for consistent inference

### Machine Learning Architecture
The modeling system implements a comprehensive ML pipeline:

- **Multi-Model Approach**: Three complementary algorithms (Logistic Regression, Random Forest, XGBoost)
- **Hyperparameter Optimization**: GridSearchCV and RandomizedSearchCV for model tuning
- **Cross-Validation**: Robust model validation with stratified splits
- **Model Persistence**: Joblib serialization for production deployment
- **Performance Tracking**: Comprehensive metrics collection (accuracy, precision, recall, F1, ROC-AUC)

### Model Explainability System
The explainability framework uses SHAP (SHapley Additive exPlanations):

- **Feature Importance**: Global and local feature importance analysis
- **Model-Agnostic**: Works across all implemented model types
- **Visualization**: Built-in plotting for feature importance and SHAP values
- **Business Insights**: Actionable insights for customer retention strategies

### Web Application Architecture
Modern separated frontend/backend architecture with enhanced features:

#### Frontend (frontend.py)
- **React-like Interactivity**: CSS animations, hover effects, gradient backgrounds
- **Professional UI Components**: Animated metric cards, interactive notifications
- **Modern Styling**: Inter font family, glass-morphism effects, responsive design
- **Enhanced Navigation**: Live system status, progress bars, real-time indicators
- **Multi-page Structure**: Dashboard, Data Explorer, AI Predictions, Performance Analytics
- **Professional Branding**: "Powered by Manmohan Mishra" with gradient styling

#### Backend (backend.py)
- **Service-Oriented Design**: Separated business logic for scalability
- **API-Ready Structure**: Prepared for REST API deployment
- **Model Management**: Training, loading, and prediction services
- **Business Intelligence**: Automated insights and recommendations generation
- **Error Resilience**: Comprehensive error handling and graceful degradation
- **Data Pipeline**: Robust preprocessing and feature engineering

#### Deployment Options
- **Single Command Deployment**: `streamlit run frontend.py --server.port 5000`
- **Production Ready**: Optimized for cloud deployment with proper configuration
- **Environment Flexibility**: Works in development, staging, and production environments

### Evaluation and Monitoring
Comprehensive model evaluation system:

- **Multi-Metric Assessment**: Standard classification metrics plus business-relevant measures
- **Confusion Matrix Analysis**: Detailed error analysis with visualization
- **ROC Curve Analysis**: Performance visualization across decision thresholds
- **Model Comparison**: Automated best model selection based on configurable criteria
- **Results Persistence**: Evaluation results saved for model tracking

## External Dependencies

### Core ML Stack
- **scikit-learn**: Primary machine learning framework for preprocessing, modeling, and evaluation
- **XGBoost**: Advanced gradient boosting implementation for ensemble modeling
- **pandas**: Data manipulation and analysis framework
- **numpy**: Numerical computing foundation

### Visualization and Analysis
- **matplotlib**: Core plotting library for data visualization
- **seaborn**: Statistical data visualization built on matplotlib
- **plotly**: Interactive plotting for web application charts

### Model Explainability
- **SHAP**: Model explainability and feature importance analysis (optional dependency with graceful fallback)

### Web Application
- **streamlit**: Web application framework for ML model deployment
- **joblib**: Model serialization and deserialization for persistence

### Development and Utilities
- **warnings**: Python standard library for managing warning filters
- **logging**: Built-in logging framework for application monitoring
- **os**: File system operations for model and data management

### Data Source
- **Telco Customer Churn Dataset**: Primary dataset for model training and evaluation
- **Fallback Sample Generation**: Synthetic data generation when primary dataset unavailable

The architecture emphasizes modularity, error resilience, and production readiness with comprehensive logging, caching strategies, and graceful degradation when optional dependencies are unavailable.