import os
import shutil
import uuid
import sys # Make sure this is imported

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# --- Import your modules ---
from .modules.cm_sdd import analyze_cross_modal # UNCOMMENT THIS LINE
# from .modules.ls_zlf import analyze_ls_zlf   # Will uncomment later
# from .modules.dp_cng import generate_counter_narrative # Will uncomment later
# --- Configuration ---
UPLOAD_DIR = "uploaded_media"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Vigilant Echo Backend",
    description="Autonomous Adversary-Aware Misinformation Counteraction System API",
    version="0.1.0"
)

# --- Static Files Setup (same as before) ---
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
    html_file_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "public", "index.html")
    if not os.path.exists(html_file_path):
        raise HTTPException(status_code=404, detail=f"Frontend HTML file not found at '{html_file_path}'. Make sure frontend/public/index.html exists.")
    with open(html_file_path, "r", encoding="utf-8") as f:
        return f.read()

# --- Analysis Endpoint (Receives Multi-Modal Files) ---
@app.post("/analyze")
async def analyze_content(
    video: UploadFile = File(None),
    audio: UploadFile = File(None), # For Round 1, this should be a text file (transcript)
    text: UploadFile = File(None)
):
    temp_files = {}

    try:
        def save_uploaded_file(uploaded_file: UploadFile, file_type: str):
            if uploaded_file:
                file_extension = os.path.splitext(uploaded_file.filename)[1]
                unique_filename = f"{uuid.uuid4()}{file_extension}"
                file_path = os.path.join(UPLOAD_DIR, unique_filename)
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(uploaded_file.file, buffer)
                temp_files[file_type] = file_path
                print(f"Saved {file_type} to: {file_path}")

        save_uploaded_file(video, "video")
        save_uploaded_file(audio, "audio") # This will be treated as text by CM-SDD
        save_uploaded_file(text, "text")

        if not temp_files:
            raise HTTPException(status_code=400, detail="No files provided for analysis. Please upload at least one file.")

        # --- Call CM-SDD ---
        print("Calling CM-SDD module...", file=sys.stderr)
        cm_sdd_results = await analyze_cross_modal(
            video_path=temp_files.get("video"),
            audio_path=temp_files.get("audio"), # Pass the path, CM-SDD will read as text
            text_path=temp_files.get("text")
        )
        print("CM-SDD module finished.", file=sys.stderr)


        # --- Placeholder for other AI Modules (LS-ZLF, DP-CNG) ---
        # These will be integrated in the next phase
        ls_zlf_results = {"status": "LS-ZLF pending", "deepfake_analysis": {"deepfake_detected": False, "reason": "Not yet analyzed"}, "llm_origin_analysis": {"llm_origin": "N/A", "confidence": 0, "reason": "Not yet analyzed"}}
        dp_cng_suggestion = "Analysis in progress. Counter-narrative will be suggested here."


        # --- Combine all results ---
        full_analysis_results = {
            "cm_sdd": cm_sdd_results,
            "ls_zlf": ls_zlf_results,
            "dp_cng_suggestion": dp_cng_suggestion,
            "overall_status": "Analysis complete with CM-SDD results."
        }

        return full_analysis_results

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during file analysis: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Internal Server Error during analysis: {str(e)}")
    finally:
        for path in temp_files.values():
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"Cleaned up temporary file: {path}")
                except OSError as e:
                    print(f"Error cleaning up file {path}: {e}", file=sys.stderr)