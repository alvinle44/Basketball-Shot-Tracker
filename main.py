from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import tempfile
import shutil
import numpy as np
import torch
from ultralytics import YOLO
from scripts.utils import get_device, smooth_point, detect_up, detect_down, score_prediction
from scripts.shot_tracker import process_video

#create fastapi instance
app = FastAPI(title="Basketball Shot Tracker API")

#ensure output folder exists 
Path("outputs").nkdir(exist_ok=True)

@app.get("/")
def home():
    return {"message": "Welcome to the Shot Tracker API"}

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Endpoint for user to upload a video such as basektball clip
    The backend will process the video and return the labeled results.
    """

    #save upload to temp file 
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    #run tracking logic 
    output_path = f"outputs/processed_{file.filename}"
    process_video(tmp_path, output_path=output_path, return_video=True)

    return FileResponse(output_path, media_type="video/mp4", filename=f"labeled_{file.filename}")

@app.get("/live")
def live_video():
    """
    Run shot tracking, but on live video 
    """
    results = process_video()
    return JSONResponse(content={
        "message": "Live Video Tracking has Ended",
        "FGM": results.get("FGM", 0),
        "FGA": results.get("FGA", 0),
        "FG%": round(results["FGM"] / results["FGA"], 2) if results["FGA"] > 0 else 0.0
    })