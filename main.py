import io
import os
import base64
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # <-- ADDED
from ultralytics import YOLO
from PIL import Image

# FIX: Prevent the "user config directory not writable" warning on Render
os.environ['YOLO_CONFIG_DIR'] = '/tmp/ultralytics'

app = FastAPI()

# --- STEP 1: ENABLE CORS ---
# This allows your HTML tester and mobile app to talk to the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all websites/local files to connect
    allow_credentials=True,
    allow_methods=["*"],  # Allows POST, GET, OPTIONS, etc.
    allow_headers=["*"],  # Allows all security headers
)

# Load YOLO model once on startup
try:
    # Render free tier has limited RAM, we load the model into CPU memory
    model = YOLO("yolov8n-seg.pt")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

PIXEL_TO_METER = 0.01  # Calibration constant

@app.get("/")
async def root():
    """Root endpoint to prevent 404 errors during health checks."""
    return {
        "status": "Online",
        "message": "Cow Weight Estimator API is running. Use the /estimate endpoint for POST requests.",
        "model_loaded": model is not None
    }

@app.post("/estimate")
async def estimate_weight_api(file: UploadFile = File(...)):
    # 1. Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # 2. Read and process image
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # 3. YOLO Segmentation Inference
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server")
        
    results = model(img_cv, imgsz=640)
    
    # Check if results and masks exist
    if not results or results[0].masks is None:
        return {"success": False, "error": "No cow detected"}

    # 4. Calculation Logic
    # results[0].masks.xy is a list of segments for detected objects
    mask_xy = results[0].masks.xy[0]
    mask_array = np.array(mask_xy)
    
    # Calculate pixel dimensions
    L_pixels = mask_array[:, 1].max() - mask_array[:, 1].min()
    G_pixels = mask_array[:, 0].max() - mask_array[:, 0].min()
    
    # Convert to meters
    L = round(L_pixels * PIXEL_TO_METER, 2)
    G = round(G_pixels * PIXEL_TO_METER, 2)
    
    # Weight formula: W = (L * G^2) / 0.3
    weight_kg = round((L * G**2) / 0.3, 2)

    # 5. Prepare Annotated Image for JSON response
    # results[0].plot() draws the masks and boxes automatically
    res_plotted = results[0].plot() 
    _, buffer = cv2.imencode('.jpg', res_plotted)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # 6. Return Clean JSON
    return {
        "success": True,
        "data": {
            "weight_kg": weight_kg,
            "length_m": L,
            "girth_m": G,
            "image_base64": img_base64
        }
    }
