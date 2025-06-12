import os
import cv2
import numpy as np
import pickle
from sklearn.naive_bayes import GaussianNB

class PlateRecognizer:
    def __init__(self, model_path):
        self.load_model(model_path)
    
    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.image_size = model_data['image_size']
        self.classes = model_data['classes']
    
    def preprocess_image(self, image):
        img_resized = cv2.resize(image, self.image_size)
        img_normalized = img_resized.astype(np.float32) / 255.0
        features = img_normalized.flatten().reshape(1, -1)
        return features
    
    def predict(self, cropped_images):
        results = []
        for img in cropped_images:
            features = self.preprocess_image(img)
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            confidence = np.max(probabilities)
            results.append((prediction, confidence))
        return results