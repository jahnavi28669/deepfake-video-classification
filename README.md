# Deep Fake Video Classification

## Overview

This repository contains code and resources for the Deep Fake Video Classification project. The objective of this project is to classify videos and images as real or fake using deep learning techniques. The model utilized is InceptionResNetV2, achieving 90% accuracy and significantly reducing manual verification time.

## Features

- **Deep Fake Classification**: Classify videos and images into real or fake categories.
- **High Accuracy**: Achieved 90% accuracy using InceptionResNetV2.
- **Efficiency**: Reduced manual verification time by 65%.
- **Technologies Used**: TensorFlow, Keras, dlib, OpenCV, FastAPI, Uvicorn.

## Requirements

- Python 3.7.12
- dlib 19.22.99
- fastapi 0.103.2
- keras 2.10.0
- opencv-python 4.10.0.84
- tensorflow 2.10.0
- uvicorn 0.22.0

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/jahnavi28669/deepfake-video-classification.git
    cd deepfake-video-classification
    ```

2. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the FastAPI server:**

   - On Windows:

     ```bash
     python -m uvicorn main:app
     ```

   - On macOS:

     ```bash
     python3 -m uvicorn main:app
     ```

2. **Open `query.html` in your web browser:**

   Navigate to `query.html` and upload your video file to get the classification results.

3. **Upload the video and get results:**

   Follow the instructions on the `query.html` page to upload your video and view the classification results.

## Project Structure

- `main.py`: The FastAPI application entry point.
- `train_model.py`: Script to train the deep learning model.
- `evaluate_model.py`: Script to evaluate model performance.
- `predict.py`: Script for making predictions on new data.
- `query.html`: HTML file for uploading videos and viewing results.
- `model/`: Directory for saving trained model files.
- `data/`: Directory for storing dataset.
- `requirements.txt`: File listing the required Python packages.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request. Ensure to follow coding standards and include tests for any new features or changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- **InceptionResNetV2**: For image classification.
- **TensorFlow and Keras**: For deep learning framework.
- **dlib**: For additional image processing functionalities.
- **OpenCV**: For computer vision tasks.
- **FastAPI and Uvicorn**: For serving models via API.
