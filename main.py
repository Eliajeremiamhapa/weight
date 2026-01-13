import io
import os
import base64
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image

os.environ['YOLO_CONFIG_DIR'] = '/tmp/ultralytics'

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = YOLO("yolov8n-seg.pt")
except Exception as e:
    model = None

PIXEL_TO_METER = 0.01 

@app.get("/")
async def root():
    return {
        "status": "Ready",
        "message": "Welcome! The Cow Weight Estimator is online and ready to help.",
        "instructions": "Send a side-view photo of your cow to the /estimate endpoint."
    }

@app.post("/estimate")
async def estimate_weight_api(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return {
            "success": False, 
            "message": "Oops! That wasn't an image. Please upload a clear JPG or PNG photo of the cow."
        }

    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        return {
            "success": False, 
            "message": "We couldn't read that image. Try taking the photo again with better lighting."
        }

    if model is None:
        return {"success": False, "message": "The AI is currently resting. Please try again in a few seconds."}
        
    results = model(img_cv, imgsz=640)
    
    if not results or results[0].masks is None:
        return {
            "success": False, 
            "message": "We couldn't find a cow in that photo. Make sure the cow is visible from the side and not blocked by anything."
        }

    mask_xy = results[0].masks.xy[0]
    mask_array = np.array(mask_xy)
    
    L_pixels = mask_array[:, 1].max() - mask_array[:, 1].min()
    G_pixels = mask_array[:, 0].max() - mask_array[:, 0].min()
    
    L = round(L_pixels * PIXEL_TO_METER, 2)
    G = round(G_pixels * PIXEL_TO_METER, 2)
    weight_kg = round((L * G**2) / 0.3, 2)

    res_plotted = results[0].plot() 
    _, buffer = cv2.imencode('.jpg', res_plotted)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # FRIENDLY RESPONSE
    return {
        "success": True,
        "message": f"Successfully calculated! Your cow is approximately {weight_kg} kg.",
        "tips": "For the best accuracy, ensure the cow is standing on level ground.",
        "data": {
            "weight_kg": weight_kg,
            "length_m": L,
            "girth_m": G,
            "image_base64": img_base64
        }
    }
