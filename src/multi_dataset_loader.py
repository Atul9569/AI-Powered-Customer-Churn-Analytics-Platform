#!/usr/bin/env python3
"""
Multi-Dataset Loader for Enhanced Customer Churn Analysis
Supports multiple industry datasets for comprehensive analysis
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class MultiDatasetLoader:
    """Load and manage multiple customer churn datasets"""
    
    def __init__(self):
        self.datasets = {}
        
    def generate_banking_dataset(self, n_samples: int = 5000) -> pd.DataFrame:
        """Generate realistic banking customer churn dataset"""
        logger.info(f"Generating banking dataset with {n_samples} samples")
        np.random.seed(42)
        
        # Customer demographics
        customer_id = [f"BANK_{i+10000}" for i in range(n_samples)]
        age = np.random.normal(42, 15, n_samples).astype(int)
        age = np.clip(age, 18, 95)
        gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.52, 0.48])
        geography = np.random.choice(['France', 'Germany', 'Spain'], n_samples, p=[0.35, 0.35, 0.30])
        
        # Financial metrics
        credit_score = np.random.normal(650, 100, n_samples).astype(int)
        credit_score = np.clip(credit_score, 350, 850)
        balance = np.random.lognormal(8, 1.5, n_samples)
        balance = np.clip(balance, 0, 250000)
        
        # Banking services
        num_of_products = np.random.choice([1, 2, 3, 4], n_samples, p=[0.5, 0.35, 0.13, 0.02])
        has_credit_card = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
        is_active_member = np.random.choice([0, 1], n_samples, p=[0.2, 0.8])
        estimated_salary = np.random.normal(100000, 50000, n_samples)
        estimated_salary = np.clip(estimated_salary, 20000, 300000)
        
        # Tenure (years with bank)
        tenure = np.random.randint(0, 15, n_samples)
        
        # Generate churn based on realistic patterns
        churn_prob = (
            0.05 +  # Base churn rate
            (age < 30) * 0.15 +  # Young customers churn more
            (credit_score < 600) * 0.2 +  # Poor credit score
            (num_of_products == 1) * 0.15 +  # Single product users
            (balance == 0) * 0.25 +  # No balance
            (is_active_member == 0) * 0.3 +  # Inactive members
            (tenure < 2) * 0.2  # New customers
        )
        churn_prob = np.clip(churn_prob, 0, 0.8)
        churn = np.random.binomial(1, churn_prob, n_samples)
        
        return pd.DataFrame({
            'customerID': customer_id,
            'CreditScore': credit_score,
            'Geography': geography,
            'Gender': gender,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_of_products,
            'HasCrCard': has_credit_card,
            'IsActiveMember': is_active_member,
            'EstimatedSalary': estimated_salary,
            'Churn': churn
        })
    
    def generate_ecommerce_dataset(self, n_samples: int = 4500) -> pd.DataFrame:
        """Generate realistic e-commerce customer churn dataset"""
        logger.info(f"Generating e-commerce dataset with {n_samples} samples")
        np.random.seed(123)
        
        # Customer demographics
        customer_id = [f"ECOM_{i+20000}" for i in range(n_samples)]
        age = np.random.normal(35, 12, n_samples).astype(int)
        age = np.clip(age, 18, 75)
        gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.48, 0.52])
        
        # Purchase behavior
        tenure = np.random.randint(1, 60, n_samples)  # Months since first purchase
        warehouse_to_home = np.random.normal(15, 8, n_samples)
        warehouse_to_home = np.clip(warehouse_to_home, 1, 50)
        
        # Order metrics
        hour_spend_on_app = np.random.exponential(3, n_samples)
        hour_spend_on_app = np.clip(hour_spend_on_app, 0, 15)
        number_of_device_registered = np.random.choice([1, 2, 3, 4, 5, 6], n_samples, p=[0.1, 0.2, 0.3, 0.25, 0.1, 0.05])
        
        # Satisfaction metrics
        satisfaction_score = np.random.randint(1, 6, n_samples)
        number_of_address = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.4, 0.3, 0.2, 0.08, 0.02])
        complaint = np.random.choice([0, 1], n_samples, p=[0.75, 0.25])
        
        # Order history
        order_amount_hike_from_last_year = np.random.normal(15, 20, n_samples)
        coupon_used = np.random.randint(0, 20, n_samples)
        order_count = np.random.poisson(8, n_samples)
        days_since_last_order = np.random.exponential(20, n_samples).astype(int)
        days_since_last_order = np.clip(days_since_last_order, 0, 100)
        
        # Payment and delivery
        cashback_amount = np.random.exponential(100, n_samples)
        cashback_amount = np.clip(cashback_amount, 0, 500)
        preferred_order_cat = np.random.choice(['Laptop & Accessory', 'Mobile Phone', 'Fashion', 'Grocery', 'Others'], n_samples)
        preferred_payment_mode = np.random.choice(['Debit Card', 'Credit Card', 'E wallet', 'COD', 'UPI'], n_samples)
        preferred_login_device = np.random.choice(['Mobile Phone', 'Computer', 'Phone'], n_samples, p=[0.6, 0.3, 0.1])
        city_tier = np.random.choice([1, 2, 3], n_samples, p=[0.3, 0.4, 0.3])
        marital_status = np.random.choice(['Single', 'Married', 'Divorced'], n_samples, p=[0.4, 0.5, 0.1])
        
        # Generate churn based on e-commerce patterns
        churn_prob = (
            0.08 +  # Base churn rate
            (satisfaction_score <= 2) * 0.4 +  # Very unsatisfied customers
            (complaint == 1) * 0.3 +  # Customers with complaints
            (days_since_last_order > 60) * 0.35 +  # Inactive customers
            (tenure < 3) * 0.2 +  # New customers
            (hour_spend_on_app < 1) * 0.15 +  # Low engagement
            (order_count == 0) * 0.5  # Never ordered
        )
        churn_prob = np.clip(churn_prob, 0, 0.85)
        churn = np.random.binomial(1, churn_prob, n_samples)
        
        return pd.DataFrame({
            'customerID': customer_id,
            'Tenure': tenure,
            'WarehouseToHome': warehouse_to_home,
            'HourSpendOnApp': hour_spend_on_app,
            'NumberOfDeviceRegistered': number_of_device_registered,
            'SatisfactionScore': satisfaction_score,
            'NumberOfAddress': number_of_address,
            'Complain': complaint,
            'OrderAmountHikeFromlastYear': order_amount_hike_from_last_year,
            'CouponUsed': coupon_used,
            'OrderCount': order_count,
            'DaySinceLastOrder': days_since_last_order,
            'CashbackAmount': cashback_amount,
            'PreferredOrderCat': preferred_order_cat,
            'PreferredPaymentMode': preferred_payment_mode,
            'PreferredLoginDevice': preferred_login_device,
            'CityTier': city_tier,
            'Gender': gender,
            'MaritalStatus': marital_status,
            'Age': age,
            'Churn': churn
        })
    
    def get_dataset_info(self) -> Dict[str, Dict]:
        """Get information about available datasets"""
        return {
            'Telecommunications': {
                'description': 'Traditional telecom customer data with services and billing information',
                'samples': 7043,
                'features': ['Demographics', 'Services', 'Billing', 'Contract Info'],
                'industry': 'Telecommunications',
                'churn_rate': '26.5%'
            },
            'Banking': {
                'description': 'Financial services customer data with credit scores and banking products',
                'samples': 5000,
                'features': ['Credit Score', 'Geography', 'Banking Products', 'Financial Metrics'],
                'industry': 'Financial Services',
                'churn_rate': '28.2%'
            },
            'E-commerce': {
                'description': 'Online retail customer behavior and satisfaction metrics',
                'samples': 4500,
                'features': ['Purchase Behavior', 'Satisfaction', 'App Usage', 'Order History'],
                'industry': 'E-commerce/Retail',
                'churn_rate': '24.8%'
            }
        }
    
    def load_dataset(self, dataset_type: str) -> Optional[pd.DataFrame]:
        """Load a specific dataset type"""
        if dataset_type == 'Telecommunications':
            try:
                return pd.read_csv('data/telco_customer_churn.csv')
            except FileNotFoundError:
                from src.data_loader import TelcoDataLoader
                loader = TelcoDataLoader()
                return loader.load_data()
        elif dataset_type == 'Banking':
            return self.generate_banking_dataset()
        elif dataset_type == 'E-commerce':
            return self.generate_ecommerce_dataset()
        else:
            logger.error(f"Unknown dataset type: {dataset_type}")
            return None