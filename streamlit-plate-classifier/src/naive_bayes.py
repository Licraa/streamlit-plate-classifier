import cv2
import numpy as np
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pickle

class SimpleGaussianNBOCR:
    def __init__(self, image_size=(32, 32)):
        self.model = GaussianNB()
        self.image_size = image_size
        self.classes = None
        
    def load_dataset(self, dataset_path):
        """Load dataset dari folder per class"""
        X = []
        y = []
        
        print("Loading dataset...")
        print(f"Dataset path: {dataset_path}")
        
        if not os.path.exists(dataset_path):
            print(f"Error: Path {dataset_path} tidak ditemukan!")
            return None, None
        
        # Get all class folders
        class_folders = [f for f in os.listdir(dataset_path) 
                        if os.path.isdir(os.path.join(dataset_path, f))]
        
        if not class_folders:
            print("Error: Tidak ada folder class ditemukan!")
            return None, None
            
        print(f"Classes found: {sorted(class_folders)}")
        
        # Load images from each class
        for class_name in class_folders:
            class_path = os.path.join(dataset_path, class_name)
            
            # Get all image files
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            print(f"Class '{class_name}': {len(image_files)} images")
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                
                try:
                    # Load and preprocess image
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"Warning: Gagal load {img_path}")
                        continue
                    
                    # Resize dan normalize
                    img_resized = cv2.resize(img, self.image_size)
                    img_normalized = img_resized.astype(np.float32) / 255.0
                    
                    # Flatten jadi feature vector
                    features = img_normalized.flatten()
                    
                    X.append(features)
                    y.append(class_name)
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        if not X:
            print("Error: Tidak ada gambar yang berhasil diload!")
            return None, None
            
        print(f"\nDataset berhasil dimuat:")
        print(f"Total samples: {len(X)}")
        print(f"Total classes: {len(set(y))}")
        print(f"Feature dimensions: {len(X[0])}")
        
        return np.array(X), np.array(y)
    
    def train(self, X, y, test_size=0.2):
        """Training Gaussian Naive Bayes"""
        
        print(f"\n{'='*50}")
        print("TRAINING GAUSSIAN NAIVE BAYES")
        print(f"{'='*50}")
        
        # Simpan classes untuk reference
        self.classes = sorted(set(y))
        print(f"Classes: {self.classes}")
        
        # Check class distribution
        unique, counts = np.unique(y, return_counts=True)
        min_samples = np.min(counts)
        
        print(f"\nClass distribution:")
        for class_name, count in zip(unique, counts):
            print(f"  {class_name}: {count} samples")
        
        print(f"\nMinimum samples per class: {min_samples}")
        
        # Split data - handle stratification based on minimum samples
        if min_samples >= 2:
            # Use stratify if all classes have at least 2 samples
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            print("Using stratified split")
        else:
            # Use regular split if some classes have only 1 sample
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            print("Using regular split (some classes have <2 samples)")
        
        print(f"\nData split:")
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        # Training
        print("\nTraining model...")
        self.model.fit(X_train, y_train)
        print("Training completed!")
        
        # Evaluasi
        print("\nEvaluating model...")
        
        # Training accuracy
        train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        
        # Testing accuracy  
        test_pred = self.model.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"\nRESULTS:")
        print(f"Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"Testing Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        # Detailed report
        print(f"\nDETAILED CLASSIFICATION REPORT:")
        print(classification_report(y_test, test_pred))
        
        # Per-class accuracy
        print(f"\nPER-CLASS ACCURACY:")
        for class_name in self.classes:
            class_mask = y_test == class_name
            if np.sum(class_mask) > 0:
                class_acc = accuracy_score(y_test[class_mask], test_pred[class_mask])
                print(f"Class '{class_name}': {class_acc:.4f} ({class_acc*100:.2f}%)")
        
        return train_acc, test_acc
    
    def predict_single(self, image_path):
        """Predict single character"""
        try:
            # Load image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Error: Tidak bisa load {image_path}")
                return None, 0
            
            # Preprocess sama seperti training
            img_resized = cv2.resize(img, self.image_size)
            img_normalized = img_resized.astype(np.float32) / 255.0
            features = img_normalized.flatten().reshape(1, -1)
            
            # Predict
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            confidence = np.max(probabilities)
            
            return prediction, confidence
            
        except Exception as e:
            print(f"Error predicting {image_path}: {e}")
            return None, 0
    
    def predict_multiple(self, image_paths):
        """Predict multiple characters"""
        results = []
        
        print(f"\nPredicting {len(image_paths)} characters...")
        
        for i, img_path in enumerate(image_paths):
            pred, conf = self.predict_single(img_path)
            
            results.append({
                'image': os.path.basename(img_path),
                'prediction': pred,
                'confidence': conf
            })
            
            print(f"{i+1}. {os.path.basename(img_path)} -> {pred} (conf: {conf:.3f})")
        
        return results
    
    def save_model(self, filepath):
        """Save model"""
        model_data = {
            'model': self.model,
            'image_size': self.image_size,
            'classes': self.classes
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nModel saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.image_size = model_data['image_size']
        self.classes = model_data['classes']
        
        print(f"Model loaded from: {filepath}")
        print(f"Classes: {self.classes}")

# MAIN EXECUTION
if __name__ == "__main__":
    
    # Inisialisasi model
    ocr = SimpleGaussianNBOCR(image_size=(32, 32))
    
    # Path ke dataset kamu
    dataset_path = "/content/custom_data03/cropped_by_character03"
    
    print("=== GAUSSIAN NAIVE BAYES OCR TRAINING ===")
    
    try:
        # 1. Load dataset
        X, y = ocr.load_dataset(dataset_path)
        
        if X is None:
            print("Gagal load dataset. Periksa path dan struktur folder!")
            exit()
        
        # 2. Training
        train_acc, test_acc = ocr.train(X, y)
        
        # 3. Save model
        model_path = "gaussian_nb_ocr.pkl"
        ocr.save_model(model_path)
        
        print(f"\n{'='*50}")
        print("TRAINING COMPLETED!")
        print(f"Final Test Accuracy: {test_acc*100:.2f}%")
        print(f"Model saved as: {model_path}")
        print(f"{'='*50}")
        
        # 4. Test prediction (optional)
        # Uncomment baris di bawah untuk test single prediction
        # test_image = "path/to/test/image.jpg"
        # if os.path.exists(test_image):
        #     pred, conf = ocr.predict_single(test_image)
        #     print(f"\nTest prediction: {pred} (confidence: {conf:.3f})")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nPastikan:")
        print("1. Path dataset benar")
        print("2. Struktur folder: dataset/A/, dataset/B/, dataset/1/, dll")
        print("3. Ada file gambar (.jpg, .png) di setiap folder")