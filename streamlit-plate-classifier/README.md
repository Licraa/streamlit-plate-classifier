# Vehicle License Plate Classifier

This project is a Streamlit web application designed for processing vehicle license plate character classification using a Naive Bayes model. The application allows users to upload images of vehicle license plates, detects the plates, crops the characters, and recognizes them using a trained model.

## Project Structure

```
streamlit-plate-classifier
├── src
│   ├── app.py                     # Main entry point for the Streamlit application
│   ├── components
│   │   ├── __init__.py            # Initializes the components package
│   │   ├── image_input.py          # Handles image upload from the user
│   │   ├── plate_detection.py       # Detects vehicle license plates
│   │   └── plate_recognition.py     # Recognizes characters from cropped images
│   ├── utils
│   │   ├── __init__.py             # Initializes the utils package
│   │   ├── file_handler.py          # Utility functions for file operations
│   │   └── image_processor.py       # Functions for image processing tasks
│   ├── models
│   │   └── gaussian_nb_ocr.pkl      # Trained Naive Bayes model for character recognition
│   └── test_samples
│       └── sample_plates            # Sample images of license plates for testing
├── tests
│   ├── __init__.py                  # Initializes the tests package
│   ├── test_detection.py             # Unit tests for plate detection functionality
│   └── test_recognition.py           # Unit tests for plate recognition functionality
├── requirements.txt                  # Lists dependencies required for the project
├── .gitignore                        # Specifies files and directories to ignore by version control
├── .streamlit
│   └── config.toml                  # Configuration settings for the Streamlit application
└── README.md                         # Documentation for the project
```

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd streamlit-plate-classifier
   ```

2. **Install dependencies**:
   It is recommended to use a virtual environment. You can create one using `venv` or `conda`.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   Use the following command to start the Streamlit application:
   ```bash
   streamlit run src/app.py
   ```

## Usage Guidelines

- Upload an image of a vehicle license plate using the provided interface.
- The application will process the image, detect the license plate, crop the characters, and recognize them using the trained model.
- The results will be displayed on the web interface, showing the recognized characters along with their confidence scores.

## Overview of Functionality

- **Image Input**: Users can upload images of license plates.
- **Plate Detection**: The application detects license plates using segmentation methods.
- **Character Cropping**: Detected characters are cropped from the license plate image.
- **Character Recognition**: The cropped characters are recognized using a trained Naive Bayes model.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.