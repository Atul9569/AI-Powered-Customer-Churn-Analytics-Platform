"""
Exploratory Data Analysis for Customer Churn Prediction
This notebook provides comprehensive EDA for the Telco Customer Churn dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

# Configure display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def load_and_explore_data():
    """Load and perform initial data exploration"""
    print("=" * 80)
    print("CUSTOMER CHURN PREDICTION - EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    
    # Load data
    from src.data_loader import TelcoDataLoader
    
    loader = TelcoDataLoader()
    df = loader.load_data()
    
    if df is None:
        print("Error: Could not load data")
        return None
    
    print(f"\n📊 Dataset Overview:")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Basic info
    print(f"\n📋 Dataset Info:")
    print(df.info())
    
    # First few rows
    print(f"\n👀 First 5 rows:")
    print(df.head())
    
    # Statistical summary
    print(f"\n📈 Statistical Summary:")
    print(df.describe())
    
    return df

def analyze_target_variable(df):
    """Analyze the target variable (Churn)"""
    print("\n" + "="*50)
    print("TARGET VARIABLE ANALYSIS")
    print("="*50)
    
    # Churn distribution
    churn_counts = df['Churn'].value_counts()
    churn_props = df['Churn'].value_counts(normalize=True)
    
    print(f"\n📊 Churn Distribution:")
    for value, count, prop in zip(churn_counts.index, churn_counts.values, churn_props.values):
        print(f"  {value}: {count:,} ({prop:.1%})")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Count plot
    sns.countplot(data=df, x='Churn', ax=axes[0])
    axes[0].set_title('Churn Distribution (Counts)')
    axes[0].set_ylabel('Number of Customers')
    
    # Percentage plot
    churn_props.plot(kind='pie', ax=axes[1], autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Churn Distribution (Percentage)')
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    plt.show()
    
    # Class imbalance check
    imbalance_ratio = churn_props.min() / churn_props.max()
    print(f"\n⚖️ Class Balance Ratio: {imbalance_ratio:.3f}")
    if imbalance_ratio < 0.5:
        print("  ⚠️ Dataset is imbalanced - consider resampling techniques")
    else:
        print("  ✅ Dataset is relatively balanced")

def analyze_demographics(df):
    """Analyze demographic features"""
    print("\n" + "="*50)
    print("DEMOGRAPHIC ANALYSIS")
    print("="*50)
    
    # Demographic columns
    demo_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, col in enumerate(demo_cols):
        # Calculate churn rate by category
        churn_by_demo = df.groupby(col)['Churn'].apply(lambda x: (x == 'Yes').mean()).sort_values(ascending=False)
        
        print(f"\n📊 {col} - Churn Rate by Category:")
        for category, rate in churn_by_demo.items():
            count = df[df[col] == category].shape[0]
            print(f"  {category}: {rate:.1%} ({count:,} customers)")
        
        # Visualization
        sns.barplot(data=df, x=col, y=(df['Churn'] == 'Yes').astype(int), ax=axes[i])
        axes[i].set_title(f'Churn Rate by {col}')
        axes[i].set_ylabel('Churn Rate')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def analyze_tenure_charges(df):
    """Analyze tenure and charges"""
    print("\n" + "="*50)
    print("TENURE AND CHARGES ANALYSIS")
    print("="*50)
    
    # Convert TotalCharges to numeric if it's not
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Basic statistics
    print(f"\n📈 Tenure Statistics:")
    print(df['tenure'].describe())
    
    print(f"\n💰 Monthly Charges Statistics:")
    print(df['MonthlyCharges'].describe())
    
    print(f"\n💰 Total Charges Statistics:")
    print(df['TotalCharges'].describe())
    
    # Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Tenure distribution
    axes[0, 0].hist(df['tenure'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Tenure Distribution')
    axes[0, 0].set_xlabel('Tenure (months)')
    axes[0, 0].set_ylabel('Frequency')
    
    # Monthly charges distribution
    axes[0, 1].hist(df['MonthlyCharges'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Monthly Charges Distribution')
    axes[0, 1].set_xlabel('Monthly Charges ($)')
    axes[0, 1].set_ylabel('Frequency')
    
    # Total charges distribution
    axes[0, 2].hist(df['TotalCharges'].dropna(), bins=20, alpha=0.7, edgecolor='black')
    axes[0, 2].set_title('Total Charges Distribution')
    axes[0, 2].set_xlabel('Total Charges ($)')
    axes[0, 2].set_ylabel('Frequency')
    
    # Tenure vs Churn
    sns.boxplot(data=df, x='Churn', y='tenure', ax=axes[1, 0])
    axes[1, 0].set_title('Tenure vs Churn')
    
    # Monthly charges vs Churn
    sns.boxplot(data=df, x='Churn', y='MonthlyCharges', ax=axes[1, 1])
    axes[1, 1].set_title('Monthly Charges vs Churn')
    
    # Total charges vs Churn
    sns.boxplot(data=df, x='Churn', y='TotalCharges', ax=axes[1, 2])
    axes[1, 2].set_title('Total Charges vs Churn')
    
    plt.tight_layout()
    plt.show()
    
    # Statistical analysis
    print(f"\n📊 Tenure Analysis by Churn:")
    tenure_by_churn = df.groupby('Churn')['tenure'].agg(['mean', 'median', 'std'])
    print(tenure_by_churn)
    
    print(f"\n📊 Monthly Charges Analysis by Churn:")
    charges_by_churn = df.groupby('Churn')['MonthlyCharges'].agg(['mean', 'median', 'std'])
    print(charges_by_churn)

def analyze_services(df):
    """Analyze service-related features"""
    print("\n" + "="*50)
    print("SERVICES ANALYSIS")
    print("="*50)
    
    # Service columns
    service_cols = [
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    # Calculate churn rates for each service
    service_churn_rates = {}
    
    for col in service_cols:
        if col in df.columns:
            churn_rate = df.groupby(col)['Churn'].apply(lambda x: (x == 'Yes').mean())
            service_churn_rates[col] = churn_rate
            
            print(f"\n📊 {col} - Churn Rate:")
            for category, rate in churn_rate.sort_values(ascending=False).items():
                count = df[df[col] == category].shape[0]
                print(f"  {category}: {rate:.1%} ({count:,} customers)")
    
    # Visualize top services impact
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    axes = axes.ravel()
    
    for i, col in enumerate(service_cols[:9]):
        if col in df.columns:
            sns.barplot(data=df, x=col, y=(df['Churn'] == 'Yes').astype(int), ax=axes[i])
            axes[i].set_title(f'Churn Rate by {col}')
            axes[i].set_ylabel('Churn Rate')
            axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Internet service deep dive
    if 'InternetService' in df.columns:
        print(f"\n🌐 Internet Service Deep Dive:")
        internet_analysis = df.groupby(['InternetService', 'Churn']).size().unstack(fill_value=0)
        internet_churn_rates = df.groupby('InternetService')['Churn'].apply(lambda x: (x == 'Yes').mean())
        
        print("Churn rates by Internet Service:")
        for service, rate in internet_churn_rates.sort_values(ascending=False).items():
            print(f"  {service}: {rate:.1%}")

def analyze_contract_billing(df):
    """Analyze contract and billing features"""
    print("\n" + "="*50)
    print("CONTRACT AND BILLING ANALYSIS")
    print("="*50)
    
    # Contract and billing columns
    contract_billing_cols = ['Contract', 'PaperlessBilling', 'PaymentMethod']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, col in enumerate(contract_billing_cols):
        if col in df.columns:
            # Calculate churn rate
            churn_rate = df.groupby(col)['Churn'].apply(lambda x: (x == 'Yes').mean()).sort_values(ascending=False)
            
            print(f"\n📊 {col} - Churn Rate:")
            for category, rate in churn_rate.items():
                count = df[df[col] == category].shape[0]
                print(f"  {category}: {rate:.1%} ({count:,} customers)")
            
            # Visualization
            sns.barplot(data=df, x=col, y=(df['Churn'] == 'Yes').astype(int), ax=axes[i])
            axes[i].set_title(f'Churn Rate by {col}')
            axes[i].set_ylabel('Churn Rate')
            axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def correlation_analysis(df):
    """Perform correlation analysis"""
    print("\n" + "="*50)
    print("CORRELATION ANALYSIS")
    print("="*50)
    
    # Prepare data for correlation
    df_encoded = df.copy()
    
    # Encode categorical variables for correlation
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    for col in categorical_cols:
        if col != 'customerID':  # Skip customer ID
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    
    # Convert TotalCharges to numeric
    df_encoded['TotalCharges'] = pd.to_numeric(df_encoded['TotalCharges'], errors='coerce')
    
    # Calculate correlation matrix
    corr_matrix = df_encoded.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(16, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # Features most correlated with Churn
    churn_corr = corr_matrix['Churn'].abs().sort_values(ascending=False)
    print(f"\n📊 Features most correlated with Churn:")
    for feature, corr_val in churn_corr[1:11].items():  # Top 10, excluding Churn itself
        print(f"  {feature}: {corr_val:.3f}")

def missing_values_analysis(df):
    """Analyze missing values"""
    print("\n" + "="*50)
    print("MISSING VALUES ANALYSIS")
    print("="*50)
    
    # Check for missing values
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing_values.index,
        'Missing Count': missing_values.values,
        'Missing Percentage': missing_percentage.values
    }).sort_values('Missing Count', ascending=False)
    
    print(f"\n📊 Missing Values Summary:")
    print(missing_df[missing_df['Missing Count'] > 0])
    
    if missing_df['Missing Count'].sum() == 0:
        print("✅ No missing values found in the dataset!")
    else:
        # Visualize missing values
        plt.figure(figsize=(10, 6))
        missing_cols = missing_df[missing_df['Missing Count'] > 0]
        sns.barplot(data=missing_cols, y='Column', x='Missing Percentage')
        plt.title('Missing Values by Column')
        plt.xlabel('Missing Percentage (%)')
        plt.tight_layout()
        plt.show()

def generate_insights(df):
    """Generate key insights from the analysis"""
    print("\n" + "="*80)
    print("KEY INSIGHTS AND RECOMMENDATIONS")
    print("="*80)
    
    # Calculate key metrics
    total_customers = len(df)
    churned_customers = (df['Churn'] == 'Yes').sum()
    churn_rate = churned_customers / total_customers
    
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Revenue impact
    avg_monthly_charges = df['MonthlyCharges'].mean()
    monthly_revenue_lost = df[df['Churn'] == 'Yes']['MonthlyCharges'].sum()
    
    print(f"\n📊 BUSINESS IMPACT:")
    print(f"  • Total Customers: {total_customers:,}")
    print(f"  • Churned Customers: {churned_customers:,}")
    print(f"  • Overall Churn Rate: {churn_rate:.1%}")
    print(f"  • Monthly Revenue Lost: ${monthly_revenue_lost:,.2f}")
    print(f"  • Average Monthly Charges: ${avg_monthly_charges:.2f}")
    
    print(f"\n🎯 KEY FINDINGS:")
    
    # Tenure insights
    avg_tenure_churned = df[df['Churn'] == 'Yes']['tenure'].mean()
    avg_tenure_retained = df[df['Churn'] == 'No']['tenure'].mean()
    print(f"  • Churned customers have {avg_tenure_retained - avg_tenure_churned:.1f} months less tenure on average")
    
    # Contract insights
    if 'Contract' in df.columns:
        month_to_month_churn = df[df['Contract'] == 'Month-to-month']['Churn'].apply(lambda x: x == 'Yes').mean()
        print(f"  • Month-to-month customers have {month_to_month_churn:.1%} churn rate")
    
    # Payment method insights
    if 'PaymentMethod' in df.columns:
        payment_churn = df.groupby('PaymentMethod')['Churn'].apply(lambda x: (x == 'Yes').mean())
        highest_churn_payment = payment_churn.idxmax()
        print(f"  • {highest_churn_payment} has the highest churn rate at {payment_churn.max():.1%}")
    
    # Service insights
    if 'InternetService' in df.columns:
        fiber_churn = df[df['InternetService'] == 'Fiber optic']['Churn'].apply(lambda x: x == 'Yes').mean()
        print(f"  • Fiber optic customers have {fiber_churn:.1%} churn rate")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"  1. Focus retention efforts on month-to-month contract customers")
    print(f"  2. Investigate fiber optic service quality issues")
    print(f"  3. Promote longer-term contracts with incentives")
    print(f"  4. Improve payment experience for electronic check users")
    print(f"  5. Implement early warning systems for customers with low tenure")
    print(f"  6. Consider loyalty programs for high-value customers")

def main():
    """Main EDA function"""
    # Load and explore data
    df = load_and_explore_data()
    if df is None:
        return
    
    # Perform various analyses
    analyze_target_variable(df)
    analyze_demographics(df)
    analyze_tenure_charges(df)
    analyze_services(df)
    analyze_contract_billing(df)
    correlation_analysis(df)
    missing_values_analysis(df)
    generate_insights(df)
    
    print(f"\n✅ Exploratory Data Analysis completed!")
    print(f"📁 Check the plots generated above for detailed visualizations")

if __name__ == "__main__":
    main()
