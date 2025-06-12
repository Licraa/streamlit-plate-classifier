def save_cropped_image(image, output_path):
    """Save the cropped image to the specified path."""
    cv2.imwrite(output_path, image)

def load_model(model_path):
    """Load the trained Naive Bayes model from the specified path."""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['image_size'], model_data['classes']