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
Path("outputs").mkdir(exist_ok=True)

@app.get("/")
def home():
    return {"message": "Welcome to the Shot Tracker API"}

#when user wants to download annotated video 
@app.get("/download/{filename}")
def download_file(filename:str):
    #check if output filepath is true by extracting the filename protion in the url 
    file_path =Path(f"outputs/processed_{filename}")
    if file_path.exists():
        #return the media file if the output is found
        return FileResponse(file_path, media_type="video/mp4", filename=filename)
    #if not found return with json reponse 
    return JSONResponse(content={
        "error":"content not found"
    }, status_code=404)

#for asynch videos want to post to count shot atttempts user uploads videos through upload request
@app.post("/upload")
#get filename from upload file 
async def upload_video(file: UploadFile = File(...), draw: bool=False):
    """
    Endpoint for user to upload a video such as basektball clip
    The backend will process the video and return the labeled results.
    """

    #save upload to temp file 
    #because cannot directly pass the upload stream to opencv downlaod as mp4 file to read 
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    file.file.close() 
    #run tracking logic 
    output_path = f"outputs/processed_{file.filename}"
    #get results from processing 
    results = process_video(tmp_path, output_path=output_path, return_video=draw)
    #return the contents from the video analysis and provide the download irl 
    return JSONResponse(content={
        "message":f"Processed {file.filename}",
        "download_url": f"http://127.0.0.1:8000/download/{file.filename}",
        "FGM": results['FGM'],
        "FGA": results['FGA'],
        "FG%": round(results["FGM"] / results["FGA"], 2) if results["FGA"] > 0 else 0.0

    })

@app.get("/live")
def live_video():
    """
    Run shot tracking, but on live video 
    """
    #live videos no annotation 
    #get live session updates of shot counts 
    results = process_video()
    return JSONResponse(content={
        "message": "Live Video Tracking has Ended",
        "FGM": results.get("FGM", 0),
        "FGA": results.get("FGA", 0),
        "FG%": round(results["FGM"] / results["FGA"], 2) if results["FGA"] > 0 else 0.0
    })