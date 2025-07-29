from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil
import uuid # To generate unique filenames

app = FastAPI()

# --- Configuration ---
# Directory where frontend static files (like index.html, style.css) will be located.
# Team 3 will put their public frontend files here.
FRONTEND_STATIC_DIR = "frontend/public"

# Directory where uploaded media files will be stored temporarily.
UPLOAD_DIR = "uploaded_media"

# Create the upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Static File Serving ---
# Mount static files for the frontend.
# This makes files in 'frontend/public' accessible via '/static/' URL path.
# For example, 'frontend/public/style.css' will be at 'http://localhost:8000/static/style.css'
# Note: This will initially show an error if 'frontend/public' doesn't exist yet,
# but it's okay, Team 3 will create it.
try:
    app.mount("/static", StaticFiles(directory=FRONTEND_STATIC_DIR), name="static")
except RuntimeError as e:
    print(f"Warning: Could not mount static files directory '{FRONTEND_STATIC_DIR}'. "
          "This is expected if the frontend setup is not complete yet. Error: {e}")


# --- Root Endpoint (Serves the main HTML page) ---
# When a user navigates to http://localhost:8000/, this function will return index.html.
@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_file_path = os.path.join(FRONTEND_STATIC_DIR, "index.html")
    if not os.path.exists(html_file_path):
        # Provide a basic placeholder if index.html isn't ready yet
        return HTMLResponse(content="""
            <!DOCTYPE html>
            <html>
            <head><title>Vigilant Echo Backend</title></head>
            <body>
                <h1>Vigilant Echo Backend Running!</h1>
                <p>Frontend (index.html) not found in 'frontend/public' yet.</p>
                <p>Waiting for Team 3 to set up the frontend.</p>
                <p>You can test the file upload endpoint by sending POST requests to /analyze.</p>
            </body>
            </html>
        """, status_code=200)
    with open(html_file_path, "r", encoding="utf-8") as f:
        return f.read()

# --- Analysis/Upload Endpoint ---
# This endpoint accepts file uploads (video, audio, text) and will later trigger analysis.
# For Hour 1, it just saves files and returns a success message.
@app.post("/analyze") # Changed from /upload to /analyze as discussed for sprint
async def analyze_content(
    video: UploadFile = File(None), # Optional video file
    audio: UploadFile = File(None), # Optional audio file
    text: UploadFile = File(None)   # Optional text file
):
    """
    Receives multi-modal files, saves them temporarily, and prepares for analysis.
    In later steps, this will call the AI modules.
    """
    
    temp_files = {} # Dictionary to store paths of saved temporary files

    try:
        # Process and save video file if provided
        if video:
            # Generate a unique filename to prevent overwrites
            file_extension = os.path.splitext(video.filename)[1]
            unique_filename = f"{uuid.uuid4()}_video{file_extension}"
            video_path = os.path.join(UPLOAD_DIR, unique_filename)
            
            # Save the file content to disk
            with open(video_path, "wb") as buffer:
                shutil.copyfileobj(video.file, buffer)
            temp_files["video"] = video_path # Store path in dictionary
            print(f"Saved video: {video_path}") # For debugging

        # Process and save audio file if provided
        if audio:
            file_extension = os.path.splitext(audio.filename)[1]
            unique_filename = f"{uuid.uuid4()}_audio{file_extension}"
            audio_path = os.path.join(UPLOAD_DIR, unique_filename)
            with open(audio_path, "wb") as buffer:
                shutil.copyfileobj(audio.file, buffer)
            temp_files["audio"] = audio_path
            print(f"Saved audio: {audio_path}") # For debugging

        # Process and save text file if provided
        if text:
            file_extension = os.path.splitext(text.filename)[1]
            unique_filename = f"{uuid.uuid4()}_text{file_extension}"
            text_path = os.path.join(UPLOAD_DIR, unique_filename)
            with open(text_path, "wb") as buffer:
                shutil.copyfileobj(text.file, buffer)
            temp_files["text"] = text_path
            print(f"Saved text: {text_path}") # For debugging
            
        # If no files were uploaded, raise an HTTP 400 error
        if not temp_files:
            raise HTTPException(status_code=400, detail="No files provided for analysis.")

        # --- Placeholder for AI Module Calls (To be added in later steps) ---
        # For Hour 1, we just return a success message.
        # In Hour 2-3, this is where you'll call analyze_cross_modal etc.
        
        # This is the initial response for Hour 1
        return {
            "message": "Files uploaded successfully! Ready for analysis.",
            "files_saved": temp_files,
            "overall_status": "Waiting for AI analysis modules."
        }

    except Exception as e:
        # Catch any unexpected errors during upload
        print(f"Error during file upload: {e}") # Log error for debugging
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
    finally:
        # IMPORTANT: In later steps, after analysis, you will add code here
        # to clean up these temporary files using os.remove(path)
        # For now, we'll leave them to confirm they're being saved.
        pass