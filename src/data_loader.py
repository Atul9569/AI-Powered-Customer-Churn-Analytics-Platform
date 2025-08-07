import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TelcoDataLoader:
    """
    Data loader for Telco Customer Churn dataset
    """
    
    def __init__(self, file_path: str = 'data/telco_customer_churn.csv'):
        self.file_path = file_path
        
    def load_data(self) -> Optional[pd.DataFrame]:
        """
        Load the Telco customer churn dataset
        
        Returns:
            pd.DataFrame: Loaded dataset or None if loading fails
        """
        try:
            df = pd.read_csv(self.file_path)
            logger.info(f"Successfully loaded data with shape: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {self.file_path}")
            # Create sample data if file doesn't exist
            return self._create_sample_data()
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None
    
    def _create_sample_data(self) -> pd.DataFrame:
        """
        Create sample Telco customer churn data for demonstration
        
        Returns:
            pd.DataFrame: Sample dataset
        """
        logger.info("Creating sample Telco customer churn dataset")
        
        np.random.seed(42)
        n_samples = 7043
        
        # Demographics
        gender = np.random.choice(['Male', 'Female'], n_samples)
        senior_citizen = np.random.choice([0, 1], n_samples, p=[0.84, 0.16])
        partner = np.random.choice(['Yes', 'No'], n_samples, p=[0.52, 0.48])
        dependents = np.random.choice(['Yes', 'No'], n_samples, p=[0.30, 0.70])
        
        # Account information
        tenure = np.random.randint(0, 73, n_samples)
        
        # Services
        phone_service = np.random.choice(['Yes', 'No'], n_samples, p=[0.90, 0.10])
        
        # Internet services
        internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.34, 0.44, 0.22])
        
        # Additional services (dependent on internet service)
        online_security = []
        online_backup = []
        device_protection = []
        tech_support = []
        streaming_tv = []
        streaming_movies = []
        
        for i in range(n_samples):
            if internet_service[i] == 'No':
                online_security.append('No internet service')
                online_backup.append('No internet service')
                device_protection.append('No internet service')
                tech_support.append('No internet service')
                streaming_tv.append('No internet service')
                streaming_movies.append('No internet service')
            else:
                online_security.append(np.random.choice(['Yes', 'No'], p=[0.50, 0.50]))
                online_backup.append(np.random.choice(['Yes', 'No'], p=[0.44, 0.56]))
                device_protection.append(np.random.choice(['Yes', 'No'], p=[0.44, 0.56]))
                tech_support.append(np.random.choice(['Yes', 'No'], p=[0.49, 0.51]))
                streaming_tv.append(np.random.choice(['Yes', 'No'], p=[0.44, 0.56]))
                streaming_movies.append(np.random.choice(['Yes', 'No'], p=[0.44, 0.56]))
        
        # Multiple lines (dependent on phone service)
        multiple_lines = []
        for i in range(n_samples):
            if phone_service[i] == 'No':
                multiple_lines.append('No phone service')
            else:
                multiple_lines.append(np.random.choice(['Yes', 'No'], p=[0.42, 0.58]))
        
        # Contract and billing
        contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.21, 0.24])
        paperless_billing = np.random.choice(['Yes', 'No'], n_samples, p=[0.59, 0.41])
        payment_method = np.random.choice([
            'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
        ], n_samples, p=[0.34, 0.19, 0.22, 0.25])
        
        # Charges
        monthly_charges = np.random.uniform(18.25, 118.75, n_samples)
        total_charges = tenure * monthly_charges + np.random.normal(0, 100, n_samples)
        total_charges = np.maximum(total_charges, 0)  # Ensure non-negative
        
        # Churn (target variable)
        # Create realistic churn probabilities based on features
        churn_prob = 0.1  # Base probability
        
        # Adjust based on features
        churn_prob_array = np.full(n_samples, 0.1)
        
        # Month-to-month contracts have higher churn
        churn_prob_array[contract == 'Month-to-month'] += 0.3
        
        # Senior citizens have higher churn
        churn_prob_array[senior_citizen == 1] += 0.1
        
        # Electronic check payment method increases churn
        churn_prob_array[payment_method == 'Electronic check'] += 0.15
        
        # Fiber optic customers have higher churn
        churn_prob_array[internet_service == 'Fiber optic'] += 0.1
        
        # Higher monthly charges increase churn
        churn_prob_array[monthly_charges > 80] += 0.1
        
        # Lower tenure increases churn
        churn_prob_array[tenure < 12] += 0.2
        
        # Ensure probabilities are between 0 and 1
        churn_prob_array = np.clip(churn_prob_array, 0, 1)
        
        churn = np.random.binomial(1, churn_prob_array, n_samples)
        churn = ['Yes' if x == 1 else 'No' for x in churn]
        
        # Create DataFrame
        df = pd.DataFrame({
            'customerID': [f'customer_{i:04d}' for i in range(n_samples)],
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'Churn': churn
        })
        
        return df
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Get basic information about the dataset
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            dict: Dictionary containing dataset information
        """
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'churn_distribution': df['Churn'].value_counts().to_dict() if 'Churn' in df.columns else None
        }
        
        return info
    
    def split_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split features and target variable
        
        Args:
            df: Complete dataset
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target
        """
        if 'Churn' not in df.columns:
            raise ValueError("Target column 'Churn' not found in dataset")
        
        # Drop customerID as it's not a feature
        features = df.drop(['Churn', 'customerID'], axis=1, errors='ignore')
        target = df['Churn']
        
        return features, target

# Usage example
if __name__ == "__main__":
    loader = TelcoDataLoader()
    data = loader.load_data()
    
    if data is not None:
        print("Data loaded successfully!")
        print(f"Shape: {data.shape}")
        print(f"Columns: {data.columns.tolist()}")
        print(f"Churn distribution:\n{data['Churn'].value_counts()}")
        
        # Save the sample data
        data.to_csv('data/telco_customer_churn.csv', index=False)
        print("Sample data saved to data/telco_customer_churn.csv")
