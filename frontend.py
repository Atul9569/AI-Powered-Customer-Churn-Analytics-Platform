import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import backend modules
try:
    from src.multi_dataset_loader import MultiDatasetLoader
    from src.preprocessing import ChurnPreprocessor
except ImportError:
    # Fallback if imports fail
    MultiDatasetLoader = None
    ChurnPreprocessor = None

import joblib
import os

# Page config
st.set_page_config(
    page_title="ChurnAI Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for React-like styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeInDown 1s ease-out;
    }
    
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: slideInUp 0.6s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 12px 48px rgba(102, 126, 234, 0.4);
    }
    
    .dataset-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        border: none;
        box-shadow: 0 10px 40px rgba(240, 147, 251, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: slideInUp 0.8s ease-out;
        cursor: pointer;
    }
    
    .dataset-card:hover {
        transform: translateY(-8px) rotate(1deg);
        box-shadow: 0 15px 60px rgba(240, 147, 251, 0.4);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 40px rgba(79, 172, 254, 0.3);
        animation: pulse 2s infinite;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 0.8rem 2.5rem;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        font-size: 1rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .feature-importance-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 5px 20px rgba(168, 237, 234, 0.3);
    }
    
    .notification-badge {
        position: absolute;
        top: -5px;
        right: -5px;
        background: #e74c3c;
        color: white;
        border-radius: 50%;
        width: 20px;
        height: 20px;
        font-size: 0.7rem;
        display: flex;
        align-items: center;
        justify-content: center;
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

# Cache functions for performance
@st.cache_resource
def get_dataset_loader():
    if MultiDatasetLoader:
        return MultiDatasetLoader()
    return None

@st.cache_resource
def get_preprocessor():
    if ChurnPreprocessor:
        return ChurnPreprocessor()
    return None

def create_animated_metrics(col1, col2, col3, col4, values, labels):
    """Create animated metric cards with React-like interactivity"""
    icons = ["👥", "🤖", "🎯", "📊"]
    
    for i, (col, value, label, icon) in enumerate(zip([col1, col2, col3, col4], values, labels, icons)):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="animation-delay: {i * 0.2}s;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h2 style="margin:0; font-size: 2.5rem; font-weight: 700;">{value}</h2>
                        <p style="margin:0; font-size: 1rem; opacity: 0.9;">{label}</p>
                    </div>
                    <div style="font-size: 2rem; opacity: 0.8;">{icon}</div>
                </div>
                <div class="notification-badge">{i+1}</div>
            </div>
            """, unsafe_allow_html=True)

def show_interactive_notifications():
    """Display interactive notification banner"""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 0.8rem; border-radius: 10px; margin-bottom: 2rem; animation: slideInDown 0.5s ease-out;">
        <div style="display: flex; justify-content: space-between; align-items: center; color: white;">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <span style="font-size: 1.5rem;">🎉</span>
                <div>
                    <strong>Welcome to ChurnAI Platform!</strong>
                    <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">Advanced ML analytics across 3 industries • Real-time predictions • 16K+ customers analyzed</p>
                </div>
            </div>
            <div style="display: flex; gap: 0.5rem;">
                <span style="background: rgba(255,255,255,0.2); padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.8rem;">LIVE</span>
                <span style="background: rgba(255,255,255,0.2); padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.8rem;">NEW</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_dashboard_overview():
    """Main dashboard overview"""
    show_interactive_notifications()
    st.markdown('<h1 class="main-header">🚀 AI-Powered Customer Churn Analytics Platform</h1>', unsafe_allow_html=True)
    
    # Load dataset
    dataset_loader = get_dataset_loader()
    
    # Dataset overview
    dataset_info = {
        "Telecommunications": {
            "icon": "📱",
            "description": "Telecom customer churn analysis",
            "customers": 7043,
            "features": 19
        },
        "Banking": {
            "icon": "🏦", 
            "description": "Banking customer retention",
            "customers": 5000,
            "features": 15
        },
        "E-commerce": {
            "icon": "🛒",
            "description": "Online retail customer behavior", 
            "customers": 4500,
            "features": 12
        }
    }
    
    # Display key metrics
    st.markdown("### 📊 Platform Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    total_customers = sum([info['customers'] for info in dataset_info.values()])
    accuracy = 67.4
    roc_auc = 73.0
    industries = len(dataset_info)
    
    create_animated_metrics(
        col1, col2, col3, col4,
        [f"{total_customers:,}", f"{accuracy}%", f"{roc_auc}%", industries],
        ["Total Customers", "Best Accuracy", "ROC-AUC Score", "Industries"]
    )
    
    # Dataset selection
    st.markdown("### 🏢 Industry Datasets")
    cols = st.columns(len(dataset_info))
    
    for i, (dataset_name, info) in enumerate(dataset_info.items()):
        with cols[i]:
            st.markdown(f"""
            <div class="dataset-card">
                <h3 style="color: white; margin: 0 0 1rem 0;">{info['icon']} {dataset_name}</h3>
                <p style="color: white; opacity: 0.9; margin: 0 0 1rem 0;">{info['description']}</p>
                <div style="display: flex; justify-content: space-between; color: white;">
                    <span><strong>{info['customers']:,}</strong> customers</span>
                    <span><strong>{info['features']}</strong> features</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

def show_data_explorer():
    """Data exploration page"""
    st.markdown('<h1 class="main-header">📊 Data Explorer</h1>', unsafe_allow_html=True)
    
    dataset_loader = get_dataset_loader()
    
    # Available datasets
    dataset_options = ["Telecommunications", "Banking", "E-commerce"]
    
    # Dataset selection
    selected_dataset = st.selectbox(
        "Select Industry Dataset",
        dataset_options,
        help="Select a dataset to view its structure and characteristics"
    )
    
    # Load selected dataset
    try:
        data = dataset_loader.load_dataset(selected_dataset) if dataset_loader else None
    except:
        data = None
    
    if data is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Customers", f"{len(data):,}")
        with col2:
            st.metric("Features", f"{len(data.columns)-1}")
        with col3:
            churn_rate = (data['Churn'].sum() / len(data) * 100) if 'Churn' in data.columns else 0
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
        
        # Data preview
        st.markdown("### 📋 Data Preview")
        st.dataframe(data.head(10), use_container_width=True)
        
        # Visualizations
        st.markdown("### 📈 Data Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Churn' in data.columns:
                fig = px.pie(
                    data, 
                    names='Churn', 
                    title="Customer Churn Distribution",
                    color_discrete_sequence=['#667eea', '#f093fb']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Show correlation for numerical columns
            numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
            if len(numerical_cols) > 1:
                corr_data = data[numerical_cols].corr()
                fig = px.imshow(
                    corr_data,
                    title="Feature Correlation Matrix",
                    color_continuous_scale='RdBu'
                )
                st.plotly_chart(fig, use_container_width=True)

def show_prediction_interface():
    """Interactive prediction interface"""
    st.markdown('<h1 class="main-header">🔮 Customer Churn Prediction</h1>', unsafe_allow_html=True)
    
    st.markdown("### 📝 Enter Customer Information")
    
    # Create input form
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    
    with col2:
        tenure = st.slider("Tenure (months)", 1, 72, 12)
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
    with col3:
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    
    with col4:
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    
    col1, col2 = st.columns(2)
    with col1:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0)
    with col2:
        payment_method = st.selectbox("Payment Method", 
                                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    
    # Prediction button
    if st.button("🚀 Predict Customer Churn", use_container_width=True, type="primary"):
        # Create input dataframe
        input_data = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [monthly_charges * tenure]
        })
        
        try:
            # Load trained models and preprocessor
            
            if os.path.exists('models/xgboost_model.joblib') and os.path.exists('models/preprocessor.joblib'):
                model = joblib.load('models/xgboost_model.joblib')
                preprocessor = joblib.load('models/preprocessor.joblib')
                
                # Preprocess input
                input_processed = preprocessor.transform(input_data)
                
                # Make prediction
                prediction = model.predict(input_processed)[0]
                probability = model.predict_proba(input_processed)[0]
                
                # Display results
                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                        <h2>⚠️ High Churn Risk</h2>
                        <p style="font-size: 1.2rem; margin: 1rem 0;">
                            This customer has a <strong>{probability[1]:.1%}</strong> probability of churning
                        </p>
                        <p>Recommended actions: Engage with retention offers, improve service quality</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-card" style="background: linear-gradient(135deg, #4ade80 0%, #22d3ee 100%);">
                        <h2>✅ Low Churn Risk</h2>
                        <p style="font-size: 1.2rem; margin: 1rem 0;">
                            This customer has a <strong>{probability[0]:.1%}</strong> probability of staying
                        </p>
                        <p>Customer likely to remain loyal. Consider upselling opportunities.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show confidence breakdown
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Churn Probability", f"{probability[1]:.1%}")
                with col2:
                    st.metric("Retention Probability", f"{probability[0]:.1%}")
                    
            else:
                st.error("Models not found. Please train models first.")
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

def show_model_performance():
    """Model performance analytics"""
    st.markdown('<h1 class="main-header">📊 Model Performance Analytics</h1>', unsafe_allow_html=True)
    
    try:
        
        if os.path.exists('models/training_results.joblib'):
            results = joblib.load('models/training_results.joblib')
            
            # Performance metrics
            st.markdown("### 🎯 Model Comparison")
            
            metrics_df = pd.DataFrame(results).T
            st.dataframe(metrics_df, use_container_width=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    x=list(results.keys()),
                    y=[results[model]['accuracy'] for model in results.keys()],
                    title="Model Accuracy Comparison",
                    color_discrete_sequence=['#667eea']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    x=list(results.keys()),
                    y=[results[model]['roc_auc'] for model in results.keys()],
                    title="ROC-AUC Score Comparison",
                    color_discrete_sequence=['#f093fb']
                )
                st.plotly_chart(fig, use_container_width=True)
                
        else:
            st.warning("No model performance data found. Train models first.")
            
    except Exception as e:
        st.error(f"Error loading performance data: {str(e)}")

def create_footer():
    """Enhanced visual footer"""
    st.markdown("---")
    
    # Create footer using Streamlit components
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 15px; color: white; margin-top: 2rem;'>
        <h3 style='text-align: center; margin-bottom: 1.5rem; color: white;'>
            🚀 ChurnAI Platform
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Three columns for footer content
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1.5rem; 
                    border-radius: 10px; height: 200px;'>
            <h4 style='color: #667eea; margin-bottom: 1rem;'>🛠️ Technology</h4>
            <p style='font-size: 0.9rem;'>• Machine Learning</p>
            <p style='font-size: 0.9rem;'>• Python & Streamlit</p>
            <p style='font-size: 0.9rem;'>• XGBoost & Random Forest</p>
            <p style='font-size: 0.9rem;'>• Interactive Analytics</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: rgba(240, 147, 251, 0.1); padding: 1.5rem; 
                    border-radius: 10px; height: 200px;'>
            <h4 style='color: #f093fb; margin-bottom: 1rem;'>📊 Performance</h4>
            <p style='font-size: 0.9rem;'><strong>Accuracy:</strong> 67.4%</p>
            <p style='font-size: 0.9rem;'><strong>ROC-AUC:</strong> 73.0%</p>
            <p style='font-size: 0.9rem;'><strong>Customers:</strong> 16,543+</p>
            <p style='font-size: 0.9rem;'><strong>Industries:</strong> 3</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: rgba(79, 172, 254, 0.1); padding: 1.5rem; 
                    border-radius: 10px; height: 200px;'>
            <h4 style='color: #4facfe; margin-bottom: 1rem;'>🎯 Features</h4>
            <p style='font-size: 0.9rem;'>• Multi-industry support</p>
            <p style='font-size: 0.9rem;'>• Real-time predictions</p>
            <p style='font-size: 0.9rem;'>• Interactive visualizations</p>
            <p style='font-size: 0.9rem;'>• Business insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Bottom attribution
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
                padding: 1.5rem; text-align: center; color: white; 
                border-radius: 10px; margin-top: 1rem;'>
        <h4 style='color: #667eea; margin-bottom: 0.5rem;'>
            ⚡ Powered by Manmohan Mishra
        </h4>
        <p style='color: #94a3b8; font-size: 0.9rem; margin: 0;'>
            Advanced Machine Learning Engineering • Enterprise Customer Analytics
        </p>
        <p style='color: #64748b; font-size: 0.8rem; margin-top: 0.5rem;'>
            © 2025 ChurnAI Platform • Predictive Analytics • Customer Retention
        </p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application"""
    # Enhanced sidebar navigation
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 1rem;">
        <h1 style="color: white; margin-bottom: 0; animation: pulse 3s infinite;">🚀 ChurnAI</h1>
        <p style="color: #ccc; font-size: 0.9rem;">Advanced Analytics Platform</p>
        <div style="display: flex; justify-content: center; gap: 0.5rem; margin-top: 1rem;">
            <div style="width: 8px; height: 8px; background: #4ade80; border-radius: 50%; animation: pulse 2s infinite;"></div>
            <span style="color: #4ade80; font-size: 0.8rem;">System Online</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    pages = {
        "🏠 Dashboard": show_dashboard_overview,
        "📊 Data Explorer": show_data_explorer,
        "🔮 AI Predictions": show_prediction_interface,
        "📈 Model Performance": show_model_performance
    }
    
    selected_page = st.sidebar.selectbox("Navigate to:", list(pages.keys()))
    
    # Enhanced sidebar metrics
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Live System Status")
    
    st.sidebar.markdown("""
    <div style="background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span>🤖 Active Models</span>
            <strong style="color: #667eea;">3/3</strong>
        </div>
        <div style="width: 100%; background: rgba(102, 126, 234, 0.2); height: 4px; border-radius: 2px;">
            <div style="width: 100%; background: #667eea; height: 4px; border-radius: 2px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Execute selected page
    pages[selected_page]()
    
    # Footer
    create_footer()

if __name__ == "__main__":
    main()