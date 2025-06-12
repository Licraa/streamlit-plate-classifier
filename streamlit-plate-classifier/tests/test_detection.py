Sure, here's the contents for the file: /streamlit-plate-classifier/streamlit-plate-classifier/tests/test_detection.py

import os
import pytest
from src.components.plate_detection import PlateDetector

API_KEY = "your_api_key"  # Replace with your actual API key
MODEL_ID = "plat-kendaraan-yolo/1"  # Replace with your actual model ID

@pytest.fixture
def plate_detector():
    return PlateDetector(API_KEY, MODEL_ID)

def test_encode_image_to_base64(plate_detector):
    image_path = "tests/test_images/sample_plate.jpg"  # Replace with a valid test image path
    encoded_image = plate_detector.encode_image_to_base64(image_path)
    assert isinstance(encoded_image, str)
    assert len(encoded_image) > 0

def test_detect_plates(plate_detector):
    image_path = "tests/test_images/sample_plate.jpg"  # Replace with a valid test image path
    detections = plate_detector.detect_plates(image_path)
    assert detections is not None
    assert 'predictions' in detections
    assert isinstance(detections['predictions'], list)

def test_crop_and_save_plates(plate_detector):
    image_path = "tests/test_images/sample_plate.jpg"  # Replace with a valid test image path
    detections = plate_detector.detect_plates(image_path)
    cropped_files = plate_detector.crop_and_save_plates(image_path, detections)
    assert isinstance(cropped_files, list)
    assert len(cropped_files) > 0
    for file in cropped_files:
        assert os.path.exists(file)