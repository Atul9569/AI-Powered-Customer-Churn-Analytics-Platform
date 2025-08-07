# ChurnAI Platform - Deployment Guide

## 🚀 Quick Deployment

The project is now structured for easy deployment with separate frontend and backend files.

## 📁 Project Structure

```
ChurnAI Platform/
├── frontend.py          # Main Streamlit UI application
├── backend.py           # Backend processing logic
├── deploy.py           # Simple deployment entry point
├── app.py              # Original combined application
├── src/                # Core business logic modules
├── models/             # Trained ML models
├── data/               # Dataset storage
└── notebooks/          # Analysis notebooks
```

## 🎯 Deployment Options

### Option 1: Frontend Only (Recommended for deployment)
```bash
streamlit run frontend.py --server.port 5000
```

### Option 2: Using Deploy Script
```bash
streamlit run deploy.py --server.port 5000
```

### Option 3: Original Combined App
```bash
streamlit run app.py --server.port 5000
```

## 🌐 Platform Features

### ✅ Frontend (frontend.py)
- **Modern React-like UI** with animations and gradients
- **Interactive dashboards** with real-time metrics
- **Multi-industry dataset explorer** (Telecom, Banking, E-commerce)
- **AI prediction interface** with customer input forms
- **Model performance analytics** with visualizations
- **Professional branding** "Powered by Manmohan Mishra"
- **Responsive design** with mobile-friendly layout

### ⚙️ Backend (backend.py)
- **Data processing pipeline** with preprocessing
- **Model training capabilities** for 3 ML algorithms
- **Prediction service** with confidence scores
- **Business insights generator** with recommendations
- **Model persistence** with joblib serialization
- **Error handling** with graceful fallbacks

## 🛠️ Technical Stack

- **Frontend Framework**: Streamlit with custom CSS
- **Machine Learning**: scikit-learn, XGBoost, pandas
- **Visualization**: Plotly, matplotlib, seaborn
- **Data Processing**: pandas, numpy
- **Model Persistence**: joblib
- **Styling**: Custom CSS with Inter font family

## 📊 Current Performance

- **Best Model Accuracy**: 67.4%
- **ROC-AUC Score**: 73.0%
- **Total Customers Analyzed**: 16,543+
- **Industries Supported**: 3 (Telecom, Banking, E-commerce)
- **Active Models**: 3 (Logistic Regression, Random Forest, XGBoost)

## 🎨 UI Features

- **Animated metric cards** with hover effects
- **Interactive notification banners** with live status
- **Gradient backgrounds** and glass-morphism effects
- **Progress bars** for system status
- **Real-time status indicators** with pulse animations
- **Professional footer** with technology showcase

## 🔧 Environment Requirements

```bash
# Core Dependencies
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=1.7.0
plotly>=5.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
```

## 🚀 Deployment Commands

### Local Development
```bash
# Start frontend application
streamlit run frontend.py --server.port 5000 --server.address 0.0.0.0
```

### Production Deployment
```bash
# For production with optimized settings
streamlit run frontend.py --server.port 5000 --server.address 0.0.0.0 --server.headless true
```

### Replit Deployment
```bash
# Configured for Replit environment
streamlit run frontend.py --server.port 5000
```

## 📈 Key Benefits of Separated Architecture

1. **Easy Deployment**: Single command to run the application
2. **Modular Design**: Clear separation between UI and logic
3. **Scalable**: Backend can be deployed separately as API
4. **Maintainable**: Frontend and backend can be updated independently
5. **Professional**: Enterprise-grade architecture with proper error handling

## 🎯 Next Steps for Production

1. **Environment Variables**: Add configuration for different environments
2. **Database Integration**: Connect to production databases
3. **API Gateway**: Deploy backend as REST API service
4. **Monitoring**: Add logging and performance monitoring
5. **Security**: Implement authentication and authorization
6. **Caching**: Add Redis for improved performance

---

**⚡ Powered by Manmohan Mishra**  
*Advanced Machine Learning Engineering • Enterprise Customer Analytics*