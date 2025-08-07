import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from typing import Tuple, List, Optional
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChurnPreprocessor:
    """
    Preprocessing pipeline for customer churn prediction
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.target_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='median')
        self.fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'ChurnPreprocessor':
        """
        Fit the preprocessing pipeline
        
        Args:
            X: Features DataFrame
            y: Target Series (optional)
            
        Returns:
            self: Fitted preprocessor
        """
        logger.info("Fitting preprocessing pipeline...")
        
        X_processed = X.copy()
        
        # Handle TotalCharges conversion (it might be object type)
        if 'TotalCharges' in X_processed.columns:
            X_processed['TotalCharges'] = pd.to_numeric(X_processed['TotalCharges'], errors='coerce')
        
        # Separate numerical and categorical features
        numerical_features = X_processed.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X_processed.select_dtypes(include=['object']).columns.tolist()
        
        logger.info(f"Numerical features: {numerical_features}")
        logger.info(f"Categorical features: {categorical_features}")
        
        # Handle missing values in numerical features
        if numerical_features:
            X_processed[numerical_features] = self.imputer.fit_transform(X_processed[numerical_features])
        
        # Encode categorical features
        for feature in categorical_features:
            le = LabelEncoder()
            # Handle missing values in categorical features
            X_processed[feature] = X_processed[feature].fillna('Unknown')
            X_processed[feature] = le.fit_transform(X_processed[feature])
            self.label_encoders[feature] = le
        
        # Create feature engineering
        X_processed = self._feature_engineering(X_processed)
        
        # Scale numerical features
        self.scaler.fit(X_processed)
        
        # Store feature names for consistency
        self.feature_names = X_processed.columns.tolist()
        
        # Fit target encoder if target is provided
        if y is not None:
            self.target_encoder.fit(y)
        
        self.fitted = True
        logger.info("Preprocessing pipeline fitted successfully!")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted pipeline
        
        Args:
            X: Features DataFrame
            
        Returns:
            np.ndarray: Transformed features
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        X_processed = X.copy()
        
        # Handle TotalCharges conversion
        if 'TotalCharges' in X_processed.columns:
            X_processed['TotalCharges'] = pd.to_numeric(X_processed['TotalCharges'], errors='coerce')
        
        # Separate numerical and categorical features
        numerical_features = X_processed.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X_processed.select_dtypes(include=['object']).columns.tolist()
        
        # Handle missing values in numerical features
        if numerical_features:
            X_processed[numerical_features] = self.imputer.transform(X_processed[numerical_features])
        
        # Encode categorical features
        for feature in categorical_features:
            if feature in self.label_encoders:
                # Handle missing values
                X_processed[feature] = X_processed[feature].fillna('Unknown')
                
                # Handle unseen categories
                le = self.label_encoders[feature]
                mask = X_processed[feature].isin(le.classes_)
                X_processed.loc[~mask, feature] = le.classes_[0]  # Default to first class
                
                X_processed[feature] = le.transform(X_processed[feature])
        
        # Apply feature engineering
        X_processed = self._feature_engineering(X_processed)
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(X_processed.columns)
        if missing_features:
            for feature in missing_features:
                X_processed[feature] = 0
        
        # Reorder columns to match training
        X_processed = X_processed[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X_processed)
        
        return X_scaled
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> np.ndarray:
        """
        Fit and transform in one step
        
        Args:
            X: Features DataFrame
            y: Target Series (optional)
            
        Returns:
            np.ndarray: Transformed features
        """
        return self.fit(X, y).transform(X)
    
    def transform_target(self, y: pd.Series) -> np.ndarray:
        """
        Transform target variable
        
        Args:
            y: Target Series
            
        Returns:
            np.ndarray: Encoded target
        """
        return self.target_encoder.transform(y)
    
    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """
        Inverse transform target variable
        
        Args:
            y: Encoded target
            
        Returns:
            np.ndarray: Original target values
        """
        return self.target_encoder.inverse_transform(y)
    
    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features
        
        Args:
            df: DataFrame with basic features
            
        Returns:
            pd.DataFrame: DataFrame with engineered features
        """
        df_eng = df.copy()
        
        # Create tenure groups
        if 'tenure' in df_eng.columns:
            df_eng['tenure_group'] = pd.cut(df_eng['tenure'], 
                                          bins=[0, 12, 24, 48, 72], 
                                          labels=[0, 1, 2, 3])
            df_eng['tenure_group'] = df_eng['tenure_group'].astype(float)
            # Fill any remaining NaNs
            df_eng['tenure_group'] = df_eng['tenure_group'].fillna(0)
        
        # Monthly charges per tenure
        if 'MonthlyCharges' in df_eng.columns and 'tenure' in df_eng.columns:
            df_eng['charges_per_tenure'] = df_eng['MonthlyCharges'] / (df_eng['tenure'] + 1)
            # Fill any remaining NaNs
            df_eng['charges_per_tenure'] = df_eng['charges_per_tenure'].fillna(0)
        
        # Total services count
        service_cols = [col for col in df_eng.columns if any(service in col.lower() for service in 
                       ['phone', 'internet', 'online', 'backup', 'protection', 'support', 'streaming'])]
        
        if service_cols:
            # Count of services (assuming encoded values, where 1 typically means 'Yes')
            df_eng['total_services'] = df_eng[service_cols].sum(axis=1)
            # Fill any remaining NaNs
            df_eng['total_services'] = df_eng['total_services'].fillna(0)
        
        # Contract risk (month-to-month is risky)
        if 'Contract' in df_eng.columns:
            # Assuming 0 = Month-to-month, 1 = One year, 2 = Two year after encoding
            df_eng['contract_risk'] = (df_eng['Contract'] == 0).astype(int)
        
        # Final check - fill any remaining NaNs with 0
        df_eng = df_eng.fillna(0)
        
        return df_eng
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names after preprocessing
        
        Returns:
            List[str]: Feature names
        """
        return self.feature_names if self.feature_names else []
    
    def save(self, filepath: str) -> None:
        """
        Save the fitted preprocessor
        
        Args:
            filepath: Path to save the preprocessor
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before saving")
        
        joblib.dump(self, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'ChurnPreprocessor':
        """
        Load a fitted preprocessor
        
        Args:
            filepath: Path to load the preprocessor from
            
        Returns:
            ChurnPreprocessor: Loaded preprocessor
        """
        preprocessor = joblib.load(filepath)
        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor

def prepare_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Prepare data for machine learning
    
    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Test set size
        random_state: Random state for reproducibility
        
    Returns:
        Tuple: X_train, X_test, y_train, y_test, preprocessor
    """
    logger.info("Preparing data for machine learning...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Initialize and fit preprocessor
    preprocessor = ChurnPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Transform targets
    y_train_processed = preprocessor.transform_target(y_train)
    y_test_processed = preprocessor.transform_target(y_test)
    
    logger.info(f"Training set shape: {X_train_processed.shape}")
    logger.info(f"Test set shape: {X_test_processed.shape}")
    
    return X_train_processed, X_test_processed, y_train_processed, y_test_processed, preprocessor

# Usage example
if __name__ == "__main__":
    from data_loader import TelcoDataLoader
    
    # Load data
    loader = TelcoDataLoader()
    data = loader.load_data()
    
    if data is not None:
        X, y = loader.split_features_target(data)
        
        # Prepare data
        X_train, X_test, y_train, y_test, preprocessor = prepare_data(X, y)
        
        print("Data preprocessing completed!")
        print(f"Feature names: {preprocessor.get_feature_names()}")
        
        # Save preprocessor
        import os
        os.makedirs('models', exist_ok=True)
        preprocessor.save('models/preprocessor.joblib')
