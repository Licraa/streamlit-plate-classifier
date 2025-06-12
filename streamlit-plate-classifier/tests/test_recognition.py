def test_plate_recognition():
    import os
    import pytest
    import cv2
    import numpy as np
    from src.utils.file_handler import load_model
    from src.components.plate_recognition import recognize_characters

    model_path = "src/models/gaussian_nb_ocr.pkl"
    model = load_model(model_path)

    test_images_folder = "src/test_samples/sample_plates"
    test_images = [f for f in os.listdir(test_images_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for image_name in test_images:
        image_path = os.path.join(test_images_folder, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        assert image is not None, f"Failed to load image: {image_path}"

        # Simulate cropping the image into characters
        # This should be replaced with actual cropping logic
        cropped_images = [image]  # Placeholder for actual cropped images

        predictions = recognize_characters(cropped_images, model)

        assert isinstance(predictions, list), "Predictions should be a list"
        assert len(predictions) > 0, f"No predictions made for image: {image_name}"

        for prediction in predictions:
            assert 'character' in prediction, "Prediction should contain 'character' key"
            assert 'confidence' in prediction, "Prediction should contain 'confidence' key"
            assert isinstance(prediction['confidence'], (float, np.float32)), "Confidence should be a float"