import cv2
import os
import dlib
import numpy as np
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
import uuid
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

# Load the model
model = load_model('mmodel.h5')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the input shape for the model
input_shape = (128, 128, 3)

# Initialize the face detector
detector = dlib.get_frontal_face_detector()


@app.post("/videoanalayzer")
async def embed_image(file: UploadFile = File(...), name: str = Form(...)):
    print('Hereeee')
    contents = await file.read()

    # Generate a unique filename to avoid conflicts
    unique_filename = f"{uuid.uuid4()}.mp4"
    with open(unique_filename, 'wb') as f:
        f.write(contents)

    # Open the video file
    cap = cv2.VideoCapture(unique_filename)

    # Get the frame rate of the video
    frameRate = cap.get(5)

    # List to store the predictions
    predictions_summary = []

    try:
        # Process the video frame by frame
        while cap.isOpened():
            frameId = cap.get(1)  # Current frame number
            ret, frame = cap.read()  # Read the next frame
            if not ret:
                break  # Exit the loop if there are no more frames

            # Process one frame every second
            if frameId % int(frameRate) == 0:
                # Detect faces in the frame
                face_rects, scores, idx = detector.run(frame, 0)
                for i, d in enumerate(face_rects):
                    # Get the coordinates of the bounding box
                    x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()

                    # Crop and preprocess the face image
                    crop_img = frame[y1:y2, x1:x2]
                    data = img_to_array(cv2.resize(crop_img, input_shape[:2])) / 255.0
                    data = np.expand_dims(data, axis=0)  # Add batch dimension

                    # Predict the class label
                    predictions = model.predict(data)
                    predicted_class = (predictions > 0.5).astype(int)  # Assuming binary classification

                    # Determine if the prediction is Fake or Real
                    result = 'Fake' if predicted_class[0][0] == 1 else 'Real'
                    predictions_summary.append(result)

        # Determine the overall classification of the video
        fake_count = predictions_summary.count('Fake')
        real_count = predictions_summary.count('Real')
        overall_result = 'DeepFake Detected' if fake_count > real_count else 'Real'

    except Exception as e:
        print(f"Error during processing: {e}")
        return {"error": "Video processing failed"}

    finally:
        # Release the video capture and remove the file
        cap.release()
        os.remove(unique_filename)

    return overall_result
