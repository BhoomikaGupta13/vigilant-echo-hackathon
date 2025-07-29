import os
import shutil
import uuid
import sys # Make sure this is imported

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# --- Import your modules ---
from .modules.cm_sdd import analyze_cross_modal # UNCOMMENT THIS LINE
from .modules.ls_zlf import analyze_ls_zlf   # Will uncomment later
from .modules.dp_cng import generate_counter_narrative # Will uncomment later
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
    audio: UploadFile = File(None),
    text: UploadFile = File(None)
):
    print(f"DEBUG: analyze_content called.", file=sys.stderr)
    print(f"DEBUG: video received: {video.filename if video else 'None'}", file=sys.stderr)
    print(f"DEBUG: audio received: {audio.filename if audio else 'None'}", file=sys.stderr)
    print(f"DEBUG: text received: {text.filename if text else 'None'}", file=sys.stderr)
    
    temp_files = {}

    try:
        # ... (save_uploaded_file helper and file saving logic, same as before) ...

        if not temp_files:
            raise HTTPException(status_code=400, detail="No files provided for analysis. Please upload at least one file.")

        # --- 1. Call CM-SDD ---
        print("Calling CM-SDD module...", file=sys.stderr)
        cm_sdd_results = await analyze_cross_modal(
            video_path=temp_files.get("video"),
            audio_path=temp_files.get("audio"),
            text_path=temp_files.get("text")
        )
        print("CM-SDD module finished.", file=sys.stderr)


        # --- 2. Call LS-ZLF ---
        # Pass all temp_files to LS-ZLF, as it might use text for LLM analysis or video for deepfake (conceptual)
        print("Calling LS-ZLF module...", file=sys.stderr)
        ls_zlf_results = await analyze_ls_zlf(temp_files)
        print("LS-ZLF module finished.", file=sys.stderr)


        # --- 3. Call DP-CNG ---
        # DP-CNG needs the results from previous modules to generate a context-aware response
        print("Calling DP-CNG module...", file=sys.stderr)
        dp_cng_suggestion = await generate_counter_narrative( # AWAIT HERE because generate_counter_narrative is now async
            {"cm_sdd": cm_sdd_results, "ls_zlf": ls_zlf_results}
        )
        print("DP-CNG module finished.", file=sys.stderr)


        # --- Combine all results ---
        full_analysis_results = {
            "cm_sdd": cm_sdd_results,
            "ls_zlf": ls_zlf_results,
            "dp_cng_suggestion": dp_cng_suggestion,
            "overall_status": "All analysis modules completed."
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