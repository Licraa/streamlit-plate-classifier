def resize_and_normalize(image, target_size=(32, 32)):
    """Resize and normalize the input image."""
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    return image

def preprocess_image(image_path):
    """Load and preprocess the image from the given path."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error: Cannot load image at {image_path}")
    return resize_and_normalize(image)

def load_and_preprocess_images(image_paths):
    """Load and preprocess multiple images."""
    processed_images = []
    for path in image_paths:
        processed_image = preprocess_image(path)
        processed_images.append(processed_image.flatten())
    return np.array(processed_images)