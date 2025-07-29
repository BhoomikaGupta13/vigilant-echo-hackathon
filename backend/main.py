import os
import shutil
import uuid # For generating unique filenames
import sys
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Import your modules (these will be created in later steps)
# They are commented out for now to ensure this file runs without errors initially.
# from backend.modules.cm_sdd import analyze_cross_modal
# from backend.modules.ls_zlf import analyze_ls_zlf
# from backend.modules.dp_cng import generate_counter_narrative

# --- Configuration ---
# Directory to temporarily store uploaded media files
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
# This assumes your project structure looks like:
# vigilant-echo/
# ├── backend/
# │   └── main.py
# └── frontend/
#     └── public/
#         ├── index.html
#         └── style.css
try:
    # Use os.path.join to ensure cross-platform compatibility for paths
    frontend_public_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "public")
    if not os.path.exists(frontend_public_dir):
        print(f"Warning: Frontend public directory not found at '{frontend_public_dir}'. "
              "Frontend might not load correctly. Please ensure the path is correct and the 'frontend/public' folder exists.",
              file=sys.stderr) # Use sys.stderr for warnings/errors
    app.mount("/static", StaticFiles(directory=frontend_public_dir), name="static")
except RuntimeError as e:
    # This might happen if the directory doesn't exist or other mounting issues
    print(f"Error mounting static files: {e}. Frontend might not load correctly.", file=sys.stderr)


# --- Root Endpoint (Serves Frontend HTML) ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Construct the path to index.html relative to main.py
    html_file_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "public", "index.html")
    
    if not os.path.exists(html_file_path):
        raise HTTPException(status_code=404, detail=f"Frontend HTML file not found at '{html_file_path}'. Make sure frontend/public/index.html exists.")
    
    with open(html_file_path, "r", encoding="utf-8") as f:
        return f.read()

# --- Analysis Endpoint (Receives Multi-Modal Files) ---
@app.post("/analyze")
async def analyze_content(
    video: UploadFile = File(None),
    audio: UploadFile = File(None),
    text: UploadFile = File(None)
):
    temp_files = {} # Dictionary to store paths of saved temporary files

    try:
        # Helper to save uploaded files to the UPLOAD_DIR
        def save_uploaded_file(uploaded_file: UploadFile, file_type: str):
            if uploaded_file:
                # Generate a unique filename to prevent conflicts
                file_extension = os.path.splitext(uploaded_file.filename)[1]
                unique_filename = f"{uuid.uuid4()}{file_extension}"
                file_path = os.path.join(UPLOAD_DIR, unique_filename)
                
                # Save the file content
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(uploaded_file.file, buffer)
                
                temp_files[file_type] = file_path
                print(f"Saved {file_type} to: {file_path}") # For debugging in console
            
        # Call the helper for each potential file type
        save_uploaded_file(video, "video")
        save_uploaded_file(audio, "audio")
        save_uploaded_file(text, "text")
            
        # Raise an error if no files were uploaded
        if not temp_files:
            raise HTTPException(status_code=400, detail="No files provided for analysis. Please upload at least one file.")

        # --- Placeholder for AI Module Calls ---
        # THESE WILL BE UNCOMMENTED AND REPLACED IN LATER STEPS!
        # For now, they return placeholder data to ensure the API works.

        # 1. Cross-Modal Semantic Discrepancy Detector (CM-SDD)
        # cm_sdd_results = await analyze_cross_modal(
        #     video_path=temp_files.get("video"),
        #     audio_path=temp_files.get("audio"),
        #     text_path=temp_files.get("text")
        # )
        cm_sdd_results = {"status": "CM-SDD pending", "discrepancy_detected": False, "discrepancy_reason": []}


        # 2. Latent-Space Zero-Shot Deepfake & LLM Fingerprint Analyzer (LS-ZLF)
        # ls_zlf_results = await analyze_ls_zlf(temp_files)
        ls_zlf_results = {"status": "LS-ZLF pending", "deepfake_analysis": {"deepfake_detected": False, "reason": "Not yet analyzed"}, "llm_origin_analysis": {"llm_origin": "N/A", "confidence": 0, "reason": "Not yet analyzed"}}


        # 3. Dynamic Persona-Based Counter-Narrative Generator (DP-CNG)
        # dp_cng_suggestion = generate_counter_narrative(
        #     {"cm_sdd": cm_sdd_results, "ls_zlf": ls_zlf_results} # Pass previous results
        # )
        dp_cng_suggestion = "Analysis in progress. Counter-narrative will be suggested here."


        # --- Combine all results into a single response ---
        full_analysis_results = {
            "cm_sdd": cm_sdd_results,
            "ls_zlf": ls_zlf_results,
            "dp_cng_suggestion": dp_cng_suggestion,
            "overall_status": "Files received and initial placeholders generated."
        }
        
        return full_analysis_results

    except HTTPException:
        # Re-raise HTTPExceptions as they are handled by FastAPI
        raise
    except Exception as e:
        # Catch any other unexpected errors during processing
        print(f"Error during file analysis: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr) # Print full traceback for debugging
        raise HTTPException(status_code=500, detail=f"Internal Server Error during analysis: {str(e)}")
    finally:
        # --- Clean up temporary files ---
        # This block ensures files are removed even if an error occurs.
        for path in temp_files.values():
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"Cleaned up temporary file: {path}") # For debugging in console
                except OSError as e:
                    print(f"Error cleaning up file {path}: {e}", file=sys.stderr)