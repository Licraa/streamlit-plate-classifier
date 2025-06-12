import cv2
import numpy as np
from PIL import Image
from segmentasi_plat import PlateDetector

def detect_plates(image):
    """Detect and segment license plates from image"""
    # Convert PIL Image to cv2 format
    img_array = np.array(image)
    
    # Initialize detector
    detector = PlateDetector(api_key="YOUR_API_KEY")
    
    # Create temporary path to save uploaded image
    temp_path = "temp_upload.jpg"
    Image.fromarray(img_array).save(temp_path)
    
    # Detect plates
    results = detector.process_single_image(temp_path)
    
    return results