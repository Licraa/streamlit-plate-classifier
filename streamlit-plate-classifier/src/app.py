import streamlit as st
import os
from PIL import Image
from segmentasi_plat import PlateDetector
from naive_bayes import SimpleGaussianNBOCR
from test_nb import PlateRecognizer, test_single_plate_folder
import numpy as np
import pickle
from dotenv import load_dotenv

load_dotenv()

def create_folders():
    """Create necessary folders for processing"""
    folders = ['temp', 'segmented_chars', 'models', 'training_results']
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

def train_model(training_path):
    """Train Naive Bayes model with character dataset"""
    st.write("📚 Starting model training...")
    
    ocr = SimpleGaussianNBOCR(image_size=(32, 32))
    
    try:
        # Load and train
        X, y = ocr.load_dataset(training_path)
        model_info = ocr.train(X, y, test_size=0.2)
        
        # Save model
        model_path = "models/gaussian_nb_ocr.pkl"
        model_data = {
            'model': ocr.model,
            'image_size': ocr.image_size,
            'classes': ocr.classes
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        st.success("✅ Model training completed!")
        return model_path
        
    except Exception as e:
        st.error(f"❌ Training error: {str(e)}")
        return None

def main():
    st.title("🚗 License Plate Recognition System")
    create_folders()
    
    # Sidebar
    st.sidebar.header("Processing Steps")
    step = st.sidebar.radio("Select Step:", 
                           ["1. Image Segmentation", 
                            "2. Model Training",
                            "3. Character Recognition"])
    
    if step == "1. Image Segmentation":
        st.subheader("Step 1: Image Segmentation")
        uploaded_file = st.file_uploader("Upload license plate image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Segment Characters"):
                with st.spinner("Segmenting characters..."):
                    # Save temporary image
                    temp_path = "temp/plate.jpg"
                    image.save(temp_path)
                    
                    # Initialize detector
                    detector = PlateDetector(
                        api_key=os.getenv('plat-kendaraan-yolo/1', 'B5ttEbfItKOVPgBxZLZq')
                    )
                    
                    # Process image
                    try:
                        results = detector.process_single_image(temp_path, 
                                                             output_folder="segmented_chars")
                        if results:
                            st.success(f"✅ Successfully segmented {len(results)} characters")
                            
                            # Display segmented characters
                            cols = st.columns(len(results))
                            for idx, (col, char_path) in enumerate(zip(cols, results)):
                                char_img = Image.open(char_path)
                                col.image(char_img, caption=f"Char {idx+1}")
                        else:
                            st.warning("No characters detected")
                    except Exception as e:
                        st.error(f"Segmentation error: {str(e)}")
    
    elif step == "2. Model Training":
        st.subheader("Step 2: Model Training")
        
        training_path = st.text_input("Enter path to training dataset:", "dataset")
        
        if st.button("Start Training"):
            model_path = train_model(training_path)
            if model_path:
                st.success(f"Model saved to {model_path}")
    
    elif step == "3. Character Recognition":
        st.subheader("Step 3: Character Recognition")
        
        if not os.path.exists("models/gaussian_nb_ocr.pkl"):
            st.error("⚠️ No trained model found. Please complete Step 2 first.")
            return
            
        if not os.listdir("segmented_chars"):
            st.error("⚠️ No segmented characters found. Please complete Step 1 first.")
            return
            
        if st.button("Recognize Characters"):
            try:
                # Initialize recognizer
                recognizer = PlateRecognizer("models/gaussian_nb_ocr.pkl")
                
                # Process segmented characters
                result = recognizer.recognize_plate_from_folder("segmented_chars")
                
                if result:
                    st.success(f"Recognized plate number: {result}")
                    
                    # Display detailed results
                    st.write("Character Recognition Details:")
                    for char, conf in result.items():
                        st.write(f"Character: {char}, Confidence: {conf:.2f}%")
                else:
                    st.warning("Could not recognize characters")
                    
            except Exception as e:
                st.error(f"Recognition error: {str(e)}")

if __name__ == "__main__":
    main()