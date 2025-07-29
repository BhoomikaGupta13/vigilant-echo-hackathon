import os
import shutil
import uuid
import sys

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# --- Import your modules ---
# Use correct relative imports. 'backend' is treated as a package when run from the root.
from backend.modules.cm_sdd import analyze_cross_modal
from backend.modules.ls_zlf import analyze_ls_zlf
from backend.modules.dp_cng import generate_counter_narrative

# --- Configuration ---
# Directory where uploaded media files will be stored temporarily.
# This should be at the project root level, relative to where Uvicorn is run (from root).
UPLOAD_DIR = "uploaded_media"
os.makedirs(UPLOAD_DIR, exist_ok=True) # Ensure the upload directory exists

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Vigilant Echo Backend",
    description="Autonomous Adversary-Aware Misinformation Counteraction System API",
    version="0.1.0"
)

# --- Static Files Setup ---
# Mount the frontend's 'public' directory to serve static files (HTML, CSS, JS)
# The path '../frontend/public' is relative to the 'backend' directory where main.py resides.
try:
    frontend_public_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "public")
    if not os.path.exists(frontend_public_dir):
        print(f"Warning: Frontend public directory not found at '{frontend_public_dir}'. "
              "Frontend might not load correctly. Please ensure the path is correct and the 'frontend/public' folder exists.",
              file=sys.stderr)
    app.mount("/static", StaticFiles(directory=frontend_public_dir), name="static")
except RuntimeError as e:
    print(f"Error mounting static files: {e}. Frontend might not load correctly.", file=sys.stderr)


# --- Root Endpoint (Serves Frontend HTML) ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Construct the path to index.html relative to main.py
    html_file_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "public", "index.html")
    
    if not os.path.exists(html_file_path):
        # If frontend/public/index.html doesn't exist, raise a 404
        raise HTTPException(status_code=404, detail=f"Frontend HTML file not found at '{html_file_path}'. Make sure frontend/public/index.html exists.")
    
    with open(html_file_path, "r", encoding="utf-8") as f:
        return f.read()

# --- Helper function to save uploaded files ---
# This helper is now inside main.py, as it was in the base code I provided.
# It uses the UPLOAD_DIR defined globally.
def save_uploaded_file(uploaded_file: UploadFile, upload_dir: str) -> str:
    """
    Save an uploaded file to a temporary location and return the file path.
    """
    if not uploaded_file.filename:
        raise ValueError("Uploaded file has no filename")
    
    # Generate unique filename to avoid conflicts
    file_extension = os.path.splitext(uploaded_file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(upload_dir, unique_filename)
    
    # Save the file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(uploaded_file.file, buffer)
    
    return file_path

# --- Analysis Endpoint (Receives Multi-Modal Files) ---
@app.post("/analyze")
async def analyze_content(
    video: UploadFile = File(None),
    audio: UploadFile = File(None), # For Round 1, this should be a text file (transcript)
    text: UploadFile = File(None)
):
    print(f"DEBUG: analyze_content called.", file=sys.stderr)
    print(f"DEBUG: video received: {video.filename if video else 'None'}", file=sys.stderr)
    print(f"DEBUG: audio received: {audio.filename if audio else 'None'}", file=sys.stderr)
    print(f"DEBUG: text received: {text.filename if text else 'None'}", file=sys.stderr)
    
    temp_files = {} # Dictionary to store paths of saved temporary files

    try:
        # Save uploaded files to temporary locations using the helper
        if video and video.filename:
            temp_files["video"] = save_uploaded_file(video, UPLOAD_DIR)
            print(f"DEBUG: Video saved to {temp_files['video']}", file=sys.stderr)
        
        if audio and audio.filename:
            temp_files["audio"] = save_uploaded_file(audio, UPLOAD_DIR)
            print(f"DEBUG: Audio saved to {temp_files['audio']}", file=sys.stderr)
        
        if text and text.filename:
            temp_files["text"] = save_uploaded_file(text, UPLOAD_DIR)
            print(f"DEBUG: Text saved to {temp_files['text']}", file=sys.stderr)

        if not temp_files:
            raise HTTPException(status_code=400, detail="No files provided for analysis. Please upload at least one file.")

        # --- 1. Call CM-SDD ---
        print("Calling CM-SDD module...", file=sys.stderr)
        # AWAIT analyze_cross_modal as it's an async function
        cm_sdd_results = await analyze_cross_modal(
            video_path=temp_files.get("video"),
            audio_path=temp_files.get("audio"), # Pass the path, CM-SDD will read as text
            text_path=temp_files.get("text")
        )
        print("CM-SDD module finished.", file=sys.stderr)
        # print(f"DEBUG: CM-SDD Results: {cm_sdd_results}", file=sys.stderr) # Uncomment for more debugging

        # --- 2. Call LS-ZLF ---
        # Pass all temp_files to LS-ZLF, as it might use text for LLM analysis or video for deepfake (conceptual)
        print("Calling LS-ZLF module...", file=sys.stderr)
        # AWAIT analyze_ls_zlf (assuming it's an async function or will be)
        ls_zlf_results = await analyze_ls_zlf(temp_files)
        print("LS-ZLF module finished.", file=sys.stderr)
        # print(f"DEBUG: LS-ZLF Results: {ls_zlf_results}", file=sys.stderr) # Uncomment for more debugging


        # --- 3. Call DP-CNG ---
        # DP-CNG needs the results from previous modules to generate a context-aware response
        print("Calling DP-CNG module...", file=sys.stderr)
        # AWAIT generate_counter_narrative (assuming it's an async function or will be)
        dp_cng_suggestion = await generate_counter_narrative(
            {"cm_sdd": cm_sdd_results, "ls_zlf": ls_zlf_results, "original_text_path": temp_files.get("text")} # Pass path for original text if needed by DP-CNG
        )
        print("DP-CNG module finished.", file=sys.stderr)
        # print(f"DEBUG: DP-CNG Suggestion: {dp_cng_suggestion}", file=sys.stderr) # Uncomment for more debugging


        # --- Combine all results ---
        full_analysis_results = {
            "cm_sdd": cm_sdd_results,
            "ls_zlf": ls_zlf_results,
            "dp_cng_suggestion": dp_cng_suggestion,
            "overall_status": "All analysis modules completed."
        }

        return full_analysis_results

    except HTTPException:
        raise # Re-raise HTTPExceptions as they are handled by FastAPI
    except Exception as e:
        # Catch any other unexpected errors during processing
        print(f"Error during file analysis: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr) # Print full traceback for debugging
        raise HTTPException(status_code=500, detail=f"Internal Server Error during analysis: {str(e)}")
    finally:
        # Clean up temporary files
        # This block ensures files are removed even if an error occurs.
        for path in temp_files.values():
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"Cleaned up temporary file: {path}", file=sys.stderr)
                except OSError as e:
                    print(f"Error cleaning up file {path}: {e}", file=sys.stderr)