import streamlit as st
from PIL import Image
import io

def upload_image():
    """Handle image upload in Streamlit"""
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        try:
            # Convert uploaded file to image
            image = Image.open(uploaded_file)
            return image
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            return None
    return None