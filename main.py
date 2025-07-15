from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
import tensorflow as tf
import numpy as np
import cv2
import time
import json
import logging
import os
import boto3

model = None
class_labels = None
disease_info = None

# Set up logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Lumora AI Plant Disease Detection API",
    description="API for detecting plant diseases from images.",
    version="1.0.0"
)

# Load the trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pdd.h5")
LABELS_PATH = os.path.join(BASE_DIR, "class_labels.json")
DISEASE_INFO_PATH = os.path.join(BASE_DIR, "disease_info.json")

# AWS S3 config
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET_NAME = os.getenv("BUCKET_NAME")
MODEL_KEY = os.getenv("MODEL_KEY", "pdd.h5")

s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, class_labels, disease_info

    logging.info("Downloading model from S3...")
    try:
        s3.download_file(BUCKET_NAME, MODEL_KEY, MODEL_PATH)
        logging.info(f"Downloaded model to {MODEL_PATH}")

        model = tf.keras.models.load_model(MODEL_PATH)
        logging.info("Model loaded successfully!")

        with open(LABELS_PATH, "r") as f:
            class_labels = json.load(f)

        with open(DISEASE_INFO_PATH, "r") as f:
            disease_info = json.load(f)

        logging.info("Class labels and disease info loaded successfully!")

    except Exception as e:
        logging.error(f"Error loading model or data: {e}")
        raise RuntimeError(f"Error during startup: {e}")

@app.get("/")
async def health_check():
    return {"status": "healthy", "message": "Lumora AI API is running"}

@app.post("/predict/")
async def predict(file: UploadFile = File(None)): # Default to None
    try:
        if file is None:
            raise HTTPException(status_code=400, detail="No file uploaded. Please upload an image.")

        if file.content_type is None:
            raise HTTPException(status_code=400, detail="Could not determine file type. Ensure correct request format.")

        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

        # Read image
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file. Could not process.")#

        # Step 1: Check if the image contains a significant amount of green (plant color)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([30, 40, 40])  # Lower bound of green color in HSV
        upper_green = np.array([90, 255, 255])  # Upper bound of green color in HSV
        mask = cv2.inRange(hsv, lower_green, upper_green)
        green_ratio = (cv2.countNonZero(mask) / (image.shape[0] * image.shape[1])) * 100

        if green_ratio < 15:  # Less than 15% green means it's probably not a plant
            raise HTTPException(status_code=400, detail="This does not appear to be a plant image. Please upload a valid plant image.")

        # Step 2: Preprocess image for model
        image = cv2.resize(image, (224, 224)) # Adjust to model input size
        image = image / 255.0  # Normalize
        image = np.expand_dims(image, axis=0)

        # Step 3: Make prediction
        predictions = model.predict(image)
        confidence = float(np.max(predictions)) * 100
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_labels.get(str(predicted_class_index), "Unknown")

        # Step 4: Get disease details
        disease_details = disease_info.get(predicted_class, {
            "cause": "Unknown",
            "symptoms": "Unknown",
            "treatment": "Unknown"
        })

        # Step 5: Construct response
        response = {
            "prediction": predicted_class,
            "confidence": f"{confidence:.2f}%",
            "disease_info": disease_details,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        return response

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Something went wrong. Contact support with error code: ERR500A.")
