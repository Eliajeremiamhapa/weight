import io
import base64
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
from PIL import Image

app = FastAPI()

# Load YOLO model on startup
# Using 'cpu' to fit within Render/Free tier memory limits (512MB)
try:
    model = YOLO("yolov8n-seg.pt")
except Exception as e:
    print(f"Error loading model: {e}")

PIXEL_TO_METER = 0.01  # Calibration constant

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
    results = model(img_cv, imgsz=640)
    
    if not results or len(results[0].masks.xy) == 0:
        return {"success": False, "error": "No cow detected"}

    # 4. Calculation Logic
    mask = results[0].masks.xy[0]
    mask_array = np.array(mask)
    
    # Calculate dimensions in meters
    L = round((mask_array[:, 1].max() - mask_array[:, 1].min()) * PIXEL_TO_METER, 2)
    G = round((mask_array[:, 0].max() - mask_array[:, 0].min()) * PIXEL_TO_METER, 2)
    
    # Weight formula: W = (L * G^2) / 0.3
    weight_kg = round((L * G**2) / 0.3, 2)

    # 5. Prepare Annotated Image for JSON response
    # (Optional: If you don't need the image back, remove this section to save bandwidth)
    res_plotted = results[0].plot() # Draws masks/boxes automatically
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