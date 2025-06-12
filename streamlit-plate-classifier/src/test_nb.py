import cv2
import numpy as np
import os
import pickle
from sklearn.naive_bayes import GaussianNB
import glob

class PlateRecognizer:
    def __init__(self, model_path):
        """Load trained model"""
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load model yang sudah di-training"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.image_size = model_data['image_size']
        self.classes = model_data['classes']
        
        print(f"Model loaded successfully!")
        print(f"Classes: {len(self.classes)} characters")
        print(f"Image size: {self.image_size}")
    
    def preprocess_image(self, image_path):
        """Preprocess image sama seperti training"""
        try:
            # Load image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Error: Cannot load {image_path}")
                return None
            
            # Resize dan normalize (sama seperti training)
            img_resized = cv2.resize(img, self.image_size)
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            # Flatten jadi feature vector
            features = img_normalized.flatten().reshape(1, -1)
            
            return features
            
        except Exception as e:
            print(f"Error preprocessing {image_path}: {e}")
            return None
    
    def predict_single_character(self, image_path):
        """Predict single character"""
        features = self.preprocess_image(image_path)
        if features is None:
            return None, 0
        
        # Predict
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        confidence = np.max(probabilities)
        
        # Clean prediction (remove 'character_' prefix)
        clean_prediction = prediction.replace('character_', '')
        
        return clean_prediction, confidence
    
    def recognize_plate_from_folder(self, folder_path, sort_by='name'):
        """
        Recognize plate dari folder berisi karakter tersegmentasi
        
        Args:
            folder_path: path ke folder berisi karakter
            sort_by: 'name' (urut nama file) atau 'modified' (urut waktu)
        """
        
        print(f"\n{'='*50}")
        print(f"RECOGNIZING PLATE FROM FOLDER")
        print(f"{'='*50}")
        print(f"Folder: {folder_path}")
        
        # Get all image files (modified version)
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = set()  # Use set to prevent duplicates
        
        for ext in image_extensions:
            files = glob.glob(os.path.join(folder_path, ext))
            files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
            image_files.update(files)
        
        # Convert back to list for sorting
        image_files = list(image_files)
        
        if not image_files:
            print("Error: No image files found!")
            return None
        
        # Sort files
        if sort_by == 'name':
            image_files.sort()  # Sort by filename
        elif sort_by == 'modified':
            image_files.sort(key=os.path.getmtime)  # Sort by modification time
        
        print(f"Found {len(image_files)} character images")
        
        # Predict each character
        plate_result = ""
        character_results = []
        
        print(f"\nPredicting characters:")
        print(f"{'No':<3} {'Filename':<20} {'Prediction':<10} {'Confidence':<12}")
        print("-" * 50)
        
        for i, img_path in enumerate(image_files):
            filename = os.path.basename(img_path)
            prediction, confidence = self.predict_single_character(img_path)
            
            if prediction is not None:
                plate_result += prediction  # This adds the prediction to plate_result
                character_results.append({  # This stores the same prediction in character_results
                    'position': i + 1,
                    'filename': filename,
                    'prediction': prediction,
                    'confidence': confidence,
                    'path': img_path
                })
                
                print(f"{i+1:<3} {filename:<20} {prediction:<10} {confidence:<12.3f}")
            else:
                print(f"{i+1:<3} {filename:<20} ERROR     0.000")
        
        # Final result
        print(f"\n{'='*50}")
        print(f"PLATE RECOGNITION RESULT")
        print(f"{'='*50}")
        print(f"Recognized Plate: {plate_result}")
        
        # Calculate average confidence
        if character_results:
            avg_confidence = np.mean([r['confidence'] for r in character_results])
            print(f"Average Confidence: {avg_confidence:.3f}")
            print(f"Characters Detected: {len(character_results)}")
        
        return {
            'plate_number': plate_result,
            'characters': character_results,
            'avg_confidence': avg_confidence if character_results else 0,
            'total_characters': len(character_results)
        }
    
    def batch_recognize_folders(self, base_path):
        """
        Recognize multiple plates (multiple folders)
        
        Args:
            base_path: path ke folder yang berisi folder-folder plate
        """
        
        print(f"\n{'='*60}")
        print(f"BATCH PLATE RECOGNITION")
        print(f"{'='*60}")
        
        # Get all subdirectories
        plate_folders = [f for f in os.listdir(base_path) 
                        if os.path.isdir(os.path.join(base_path, f))]
        
        if not plate_folders:
            print("No plate folders found!")
            return []
        
        plate_folders.sort()
        print(f"Found {len(plate_folders)} plate folders")
        
        results = []
        
        for folder_name in plate_folders:
            folder_path = os.path.join(base_path, folder_name)
            print(f"\nProcessing: {folder_name}")
            
            result = self.recognize_plate_from_folder(folder_path)
            if result:
                result['folder_name'] = folder_name
                results.append(result)
                print(f"Result: {result['plate_number']}")
        
        # Summary
        print(f"\n{'='*60}")
        print(f"BATCH RECOGNITION SUMMARY")
        print(f"{'='*60}")
        print(f"{'Folder':<20} {'Plate Number':<15} {'Confidence':<12} {'Chars'}")
        print("-" * 60)
        
        for result in results:
            print(f"{result['folder_name']:<20} {result['plate_number']:<15} "
                  f"{result['avg_confidence']:<12.3f} {result['total_characters']}")
        
        return results

def calculate_accuracy(predicted_plate, ground_truth):
    """Calculate accuracy between predicted and actual plate number"""
    if len(predicted_plate) != len(ground_truth):
        return 0.0
    
    correct = sum(1 for p, t in zip(predicted_plate, ground_truth) if p == t)
    return (correct / len(ground_truth)) * 100

# MAIN TESTING FUNCTIONS
def test_single_plate_folder(model_path, plate_folder_path, ground_truth):
    """Test single plate folder with accuracy calculation"""
    
    # Load model
    recognizer = PlateRecognizer(model_path)
    
    # Recognize plate
    result = recognizer.recognize_plate_from_folder(plate_folder_path)
    
    if result:
        # Calculate accuracy
        accuracy = calculate_accuracy(result['plate_number'], ground_truth)
        result['accuracy'] = accuracy
        
        # Print comparison table
        print("\n" + "="*60)
        print("RECOGNITION ACCURACY COMPARISON")
        print("="*60)
        print(f"{'Metric':<20} {'Value':<20}")
        print("-"*60)
        print(f"{'Ground Truth':<20} {ground_truth:<20}")
        print(f"{'Predicted':<20} {result['plate_number']:<20}")
        print(f"{'Accuracy':<20} {accuracy:,.2f}%")
        print(f"{'Confidence':<20} {result['avg_confidence']*100:,.2f}%")
        print("="*60)
    
    return result

# Modify main section
if __name__ == "__main__":
    model_path = "gaussian_nb_ocr.pkl"
    
    # Test cases with ground truth
    test_cases = [
        {
            "folder": "Data_Testing/BP_2565_WP",
            "ground_truth": "BP2565WP",
            # "ground_truth": "H4157K",
            # "ground_truth": "H6426ALC",
            "method": "Naive Bayes"
        },
        # Add more test cases here
    ]
    
    # Results storage
    comparison_results = []
    
    for test in test_cases:
        print(f"\n=== TESTING {test['method']} ===")
        if os.path.exists(test['folder']):
            result = test_single_plate_folder(model_path, test['folder'], test['ground_truth'])
            if result:
                comparison_results.append({
                    'method': test['method'],
                    'predicted': result['plate_number'],
                    'ground_truth': test['ground_truth'],
                    'accuracy': result.get('accuracy', 0),
                    'confidence': result['avg_confidence'] * 100
                })
        else:
            print(f"Folder {test['folder']} tidak ditemukan!")

    # Print final comparison table for all methods
    if comparison_results:
        print("\n" + "="*80)
        print("METHODS COMPARISON TABLE")
        print("="*80)
        print(f"{'Method':<15} {'Ground Truth':<15} {'Predicted':<15} {'Accuracy':<10} {'Confidence':<10}")
        print("-"*80)
        
        for res in comparison_results:
            print(f"{res['method']:<15} {res['ground_truth']:<15} {res['predicted']:<15} "
                  f"{res['accuracy']:,.2f}% {res['confidence']:,.2f}%")
        print("="*80)