"""
Deployment configuration for ChurnAI Platform
Simplified entry point for easy deployment
"""

import streamlit as st
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the frontend application
from frontend import main

# Page configuration
st.set_page_config(
    page_title="ChurnAI Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

if __name__ == "__main__":
    # Run the main application
    main()