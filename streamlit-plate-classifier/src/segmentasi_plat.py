import os
import cv2
import requests
import base64
import numpy as np
from PIL import Image
import json

class PlateDetector:
    def __init__(self, api_key, model_id="plat-kendaraan-yolo/1"):
        """
        Inisialisasi detector plat kendaraan
        
        Args:
            api_key (str): API key Roboflow Anda
            model_id (str): ID model yang akan digunakan (format: workspace/model-name/version)
        """
        self.api_key = api_key
        self.model_id = model_id
        self.api_url = f"https://detect.roboflow.com/{model_id}"
        print(f"🔧 Menggunakan model: {model_id}")
        print(f"🌐 API URL: {self.api_url}")
    
    def encode_image_to_base64(self, image_path):
        """Encode gambar ke base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def detect_plates(self, image_path, confidence_threshold=0.4, overlap_threshold=0.3):
        """
        Deteksi plat kendaraan menggunakan API Roboflow
        
        Args:
            image_path (str): Path ke gambar
            confidence_threshold (float): Minimum confidence
            overlap_threshold (float): Threshold untuk NMS
            
        Returns:
            dict: Hasil deteksi dari API
        """
        try:
            # Method 1: POST dengan base64 (lebih universal)
            print(f"📤 Mengirim request ke API untuk gambar: {os.path.basename(image_path)}")
            
            # Encode gambar ke base64
            encoded_image = self.encode_image_to_base64(image_path)
            
            # Siapkan URL dengan parameter
            params = {
                "api_key": self.api_key,
                "confidence": int(confidence_threshold * 100),  # Convert to percentage
                "overlap": int(overlap_threshold * 100),
                "format": "json"
            }
            
            # Siapkan data untuk POST request
            data = {
                "image": encoded_image
            }
            
            # Kirim POST request
            response = requests.post(self.api_url, params=params, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Berhasil mendapat response dari API")
                return result
            else:
                print(f"❌ Error API: {response.status_code}")
                print(f"Response: {response.text[:200]}...")
                
                # Coba method alternatif dengan multipart form
                return self._detect_plates_multipart(image_path, confidence_threshold, overlap_threshold)
                
        except requests.exceptions.Timeout:
            print("⏰ Request timeout - coba lagi dengan gambar yang lebih kecil")
            return None
        except Exception as e:
            print(f"❌ Error saat deteksi: {str(e)}")
            return None
    
    def _detect_plates_multipart(self, image_path, confidence_threshold=0.4, overlap_threshold=0.3):
        """
        Method alternatif menggunakan multipart form data
        """
        try:
            print("🔄 Mencoba method alternatif (multipart form)...")
            
            # Siapkan URL dengan parameter
            params = {
                "api_key": self.api_key,
                "confidence": int(confidence_threshold * 100),
                "overlap": int(overlap_threshold * 100)
            }
            
            # Siapkan file untuk upload
            with open(image_path, 'rb') as image_file:
                files = {"file": image_file}
                response = requests.post(self.api_url, params=params, files=files, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Berhasil dengan method alternatif")
                return result
            else:
                print(f"❌ Method alternatif juga gagal: {response.status_code}")
                print(f"Response: {response.text[:200]}...")
                return None
                
        except Exception as e:
            print(f"❌ Error method alternatif: {str(e)}")
            return None
    
    def create_output_folder(self, folder_name="cropped_plates"):
        """Membuat folder output jika belum ada"""
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"Folder '{folder_name}' berhasil dibuat")
        return folder_name
    
    def crop_and_save_plates(self, image_path, detections, output_folder="cropped_plates"):
        """
        Crop dan simpan plat yang terdeteksi
        
        Args:
            image_path (str): Path gambar original
            detections (dict): Hasil deteksi dari API
            output_folder (str): Folder output
            
        Returns:
            list: List path file yang berhasil di-crop
        """
        if not detections or 'predictions' not in detections:
            print("Tidak ada deteksi plat yang ditemukan")
            return []
        
        # Buat folder output
        output_folder = self.create_output_folder(output_folder)
        
        # Load gambar original
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Tidak dapat membaca gambar {image_path}")
            return []
        
        height, width = image.shape[:2]
        cropped_files = []
        
        predictions = detections['predictions']
        print(f"Ditemukan {len(predictions)} deteksi plat")
        
        for i, prediction in enumerate(predictions):
            try:
                confidence = prediction.get('confidence', 0)
                class_name = prediction.get('class', 'plate')
                
                # Ambil koordinat bounding box
                x_center = prediction['x']
                y_center = prediction['y']
                box_width = prediction['width']
                box_height = prediction['height']
                
                # Konversi ke koordinat corner
                x1 = int(x_center - box_width / 2)
                y1 = int(y_center - box_height / 2)
                x2 = int(x_center + box_width / 2)
                y2 = int(y_center + box_height / 2)
                
                # Pastikan koordinat dalam batas gambar
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)
                
                # Tambahkan padding kecil (opsional)
                padding = 5
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(width, x2 + padding)
                y2 = min(height, y2 + padding)
                
                # Crop gambar
                cropped_plate = image[y1:y2, x1:x2]
                
                if cropped_plate.size > 0 and cropped_plate.shape[0] > 0 and cropped_plate.shape[1] > 0:
                    # Generate nama file output
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    output_filename = f"{base_name}_plate_{i+1}_conf{confidence:.2f}.jpg"
                    output_path = os.path.join(output_folder, output_filename)
                    
                    # Simpan hasil crop
                    success = cv2.imwrite(output_path, cropped_plate)
                    
                    if success:
                        cropped_files.append(output_path)
                        print(f"✅ Plat {i+1} berhasil disimpan: {output_filename}")
                        print(f"   - Confidence: {confidence:.2f}")
                        print(f"   - Ukuran: {cropped_plate.shape[1]}x{cropped_plate.shape[0]}")
                        print(f"   - Class: {class_name}")
                    else:
                        print(f"❌ Gagal menyimpan plat {i+1}")
                else:
                    print(f"⚠️ Plat {i+1} terlalu kecil atau kosong, diabaikan")
                    
            except Exception as e:
                print(f"Error saat memproses plat {i+1}: {str(e)}")
        
        return cropped_files
    
    def process_single_image(self, image_path, output_folder="cropped_plates", confidence_threshold=0.4):
        """
        Proses satu gambar: deteksi dan crop plat
        
        Args:
            image_path (str): Path ke gambar
            output_folder (str): Folder output
            confidence_threshold (float): Minimum confidence
            
        Returns:
            list: List file hasil crop
        """
        print(f"\n{'='*60}")
        print(f"Memproses gambar: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        
        # Deteksi plat
        detections = self.detect_plates(image_path, confidence_threshold)
        
        if detections is None:
            print("❌ Gagal mendapat hasil deteksi")
            return []
        
        # Crop dan simpan
        cropped_files = self.crop_and_save_plates(image_path, detections, output_folder)
        
        print(f"\n📊 Ringkasan:")
        print(f"   - Total deteksi: {len(detections.get('predictions', []))}")
        print(f"   - Berhasil di-crop: {len(cropped_files)}")
        
        return cropped_files
    
    def process_multiple_images(self, image_folder, output_folder="testing_dataset", 
                              confidence_threshold=0.4, supported_formats=None):
        """
        Proses multiple gambar dalam folder
        
        Args:
            image_folder (str): Path ke folder gambar
            output_folder (str): Folder output
            confidence_threshold (float): Minimum confidence
            supported_formats (list): Format file yang didukung
        """
        if supported_formats is None:
            supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        if not os.path.exists(image_folder):
            print(f"❌ Error: Folder {image_folder} tidak ditemukan")
            return
        
        # Ambil semua file gambar
        image_files = []
        for filename in os.listdir(image_folder):
            if any(filename.lower().endswith(ext) for ext in supported_formats):
                image_files.append(os.path.join(image_folder, filename))
        
        if not image_files:
            print(f"❌ Tidak ada gambar ditemukan di folder {image_folder}")
            return
        
        print(f"\n🔍 Ditemukan {len(image_files)} gambar untuk diproses")
        print(f"📁 Hasil akan disimpan di folder: {output_folder}")
        
        total_cropped = 0
        successful_images = 0
        
        for i, image_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing: {os.path.basename(image_file)}")
            try:
                cropped_files = self.process_single_image(image_file, output_folder, confidence_threshold)
                total_cropped += len(cropped_files)
                if cropped_files:
                    successful_images += 1
            except Exception as e:
                print(f"❌ Error processing {image_file}: {str(e)}")
        
        print(f"\n🎉 SELESAI!")
        print(f"{'='*60}")
        print(f"📊 Ringkasan Akhir:")
        print(f"   - Total gambar diproses: {len(image_files)}")
        print(f"   - Gambar dengan plat terdeteksi: {successful_images}")
        print(f"   - Total plat berhasil di-crop: {total_cropped}")
        print(f"   - Folder hasil: {output_folder}")
        print(f"{'='*60}")

    def validate_api_key(self):
        # Add validation logic here
        return bool(self.api_key and len(self.api_key) > 0)

# Contoh penggunaan
def main():
    # PENTING: Ganti dengan API key Anda sendiri!
    # Dapatkan API key dari: https://app.roboflow.com/settings/api
    API_KEY = "B5ttEbfItKOVPgBxZLZq"  # Ganti dengan API key Anda
    
    # Format model ID: workspace/model-name/version
    # Contoh: "your-workspace/plat-kendaraan-yolo/1"
    MODEL_ID = "plat-kendaraan-yolo/1"  # Sesuaikan dengan workspace dan versi model Anda
    
    print("🚀 Memulai Plate Detection System")
    print(f"📋 Model ID: {MODEL_ID}")
    
    # Inisialisasi detector
    detector = PlateDetector(API_KEY, MODEL_ID)
    
    # PILIHAN PENGGUNAAN:
    
    # 1️⃣ Proses satu gambar
    print("\n" + "="*50)
    print("PILIHAN 1: Proses Satu Gambar")
    print("="*50)
    
    # Uncomment baris di bawah untuk memproses satu gambar
    image_path = "H_4157_K.png"  # Ganti dengan path gambar Anda
    if os.path.exists(image_path):
        cropped_files = detector.process_single_image(
            image_path=image_path,
            output_folder="hasil_crop",
            confidence_threshold=0.5
        )
        print(f"🎯 Hasil crop tersimpan: {len(cropped_files)} file")
    else:
        print(f"❌ File {image_path} tidak ditemukan")
    
    # 2️⃣ Proses semua gambar dalam folder (AKTIF)
    # print("\n" + "="*50)
    # print("PILIHAN 2: Batch Processing")
    # print("="*50)
    
    # image_folder = "input_images"  # Ganti dengan path folder Anda
    
    # Buat folder input jika belum ada
    # if not os.path.exists(image_folder):
    #     os.makedirs(image_folder)
    #     print(f"📁 Folder '{image_folder}' telah dibuat")
    #     print(f"💡 Letakkan gambar-gambar Anda di folder '{image_folder}' dan jalankan lagi script ini")
    #     return
    
    # # Proses batch
    # detector.process_multiple_images(
    #     image_folder=image_folder,
    #     output_folder="hasil_crop",
    #     confidence_threshold=0.4,  # Sesuaikan threshold (0.1 - 0.9)
    #     supported_formats=['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    # )

# # Fungsi untuk testing koneksi API
# def test_api_connection(api_key, model_id):
#     """
#     Test koneksi ke API Roboflow
#     """
#     print("🔧 Testing API Connection...")
#     detector = PlateDetector(api_key, model_id)
    
#     # Buat gambar test sederhana (1x1 pixel)
#     import tempfile
#     import numpy as np
    
#     # Buat gambar dummy
#     dummy_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
#     temp_path = os.path.join(tempfile.gettempdir(), "test_image.jpg")
#     cv2.imwrite(temp_path, dummy_image)
    
#     try:
#         result = detector.detect_plates(temp_path, confidence_threshold=0.1)
#         if result is not None:
#             print("✅ API connection successful!")
#             print(f"📊 Response keys: {list(result.keys())}")
#             return True
#         else:
#             print("❌ API connection failed")
#             return False
#     finally:
#         # Hapus file temporary
#         if os.path.exists(temp_path):
#             os.remove(temp_path)

if __name__ == "__main__":
    # Uncomment untuk test koneksi API terlebih dahulu
    # test_api_connection("YOUR_API_KEY", "plat-kendaraan-yolo/1")
    
    main()