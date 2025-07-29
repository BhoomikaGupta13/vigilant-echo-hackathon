import os
import shutil
import uuid
import sys # Import sys for printing debug messages to stderr

from fastapi import FastAPI, UploadFile, File, HTTPException, Form # Ensure Form is imported for source_id
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Depends # For database dependency injection

# --- Import your AI modules ---
# Use correct relative imports. 'backend' is treated as a package when run from the root.
from .modules.cm_sdd import analyze_cross_modal
from .modules.ls_zlf import analyze_ls_zlf
from .modules.dp_cng import generate_counter_narrative

# --- Database Imports ---
from sqlalchemy.orm import Session # For database session type hinting
from sqlalchemy.sql import func # For database functions like 'now()'
from .db.database import Base, engine, SessionLocal, get_db # Import all necessary database components
from .db.models import Source # Import your Source model

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

# Debug function to check environment
def debug_environment():
    """Debug function to check all dependencies."""
    print("=== ENVIRONMENT DEBUG ===", file=sys.stderr)
    
    # Check FFmpeg
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
        print(f"FFmpeg: {'✓ Available' if result.returncode == 0 else '✗ Failed'}", file=sys.stderr)
    except:
        print("FFmpeg: ✗ Not found", file=sys.stderr)
    
    # Check Python libraries
    libraries = ['pydub', 'librosa', 'soundfile', 'torch', 'torchaudio', 'transformers', 'sentence_transformers']
    for lib in libraries:
        try:
            __import__(lib)
            print(f"{lib}: ✓ Available", file=sys.stderr)
        except ImportError:
            print(f"{lib}: ✗ Missing", file=sys.stderr)
    
    print("=== END DEBUG ===", file=sys.stderr)

# --- Database Initialization on Startup ---
# This function creates all defined database tables (like 'sources')
# if they don't already exist in your 'vigilant_echo.db' file.
@app.on_event("startup")
def on_startup_create_db_tables():
    print("INFO: Creating database tables...", file=sys.stderr)
    try:
        # Base.metadata.create_all(bind=engine) inspects all models inherited from Base
        # and creates their corresponding tables in the database.
        Base.metadata.create_all(bind=engine)
        print("INFO: Database tables created/checked successfully.", file=sys.stderr)
        
        # Run environment debug
        debug_environment()
        
    except Exception as e:
        print(f"ERROR: Failed to create database tables: {e}", file=sys.stderr)

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

# --- NEW: Sources Endpoint (Get All Tracked Sources) ---
@app.get("/sources")
async def get_tracked_sources(db: Session = Depends(get_db)):
    """
    Retrieve all tracked sources from the database.
    Returns a list of sources with their flag counts and other metadata.
    """
    try:
        # Query all sources from the database, ordered by flag_count descending (most flagged first)
        sources = db.query(Source).order_by(Source.flag_count.desc()).all()
        
        # Convert SQLAlchemy objects to dictionaries for JSON response
        sources_list = []
        for source in sources:
            # Handle the case where last_flagged_at might have a default timestamp
            # If flag_count is 0, we assume it was never actually flagged
            last_flagged_display = None
            if source.flag_count > 0 and source.last_flagged_at:
                last_flagged_display = source.last_flagged_at.isoformat()
            
            sources_list.append({
                "source_id": source.source_id,
                "flag_count": source.flag_count,
                "is_high_risk": source.is_high_risk,
                "created_at": source.created_at.isoformat() if source.created_at else None,
                "last_flagged_at": last_flagged_display
            })
        
        print(f"DEBUG: Retrieved {len(sources_list)} sources from database", file=sys.stderr)
        return sources_list
        
    except Exception as e:
        print(f"ERROR: Failed to retrieve sources: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Internal Server Error while fetching sources: {str(e)}")

# --- Helper function to save uploaded files ---
# This helper is a standalone function, called from analyze_content.
def save_uploaded_file_helper(uploaded_file: UploadFile, upload_dir: str) -> str:
    """
    Save an uploaded file to a temporary location and return the file path.
    """
    print(f"DEBUG: Saving uploaded file: {uploaded_file.filename}", file=sys.stderr)
    
    if not uploaded_file.filename:
        raise ValueError("Uploaded file has no filename") # Ensure filename exists
    
    file_extension = os.path.splitext(uploaded_file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(upload_dir, unique_filename)
    
    # Save the file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(uploaded_file.file, buffer)
        
        # Verify file was saved
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"DEBUG: File saved successfully: {file_path} (size: {file_size} bytes)", file=sys.stderr)
            return file_path
        else:
            raise Exception("File was not saved properly")
            
    except Exception as e:
        print(f"ERROR: Failed to save file {uploaded_file.filename}: {e}", file=sys.stderr)
        raise

# --- Analysis Endpoint (Receives Multi-Modal Files and Source ID) ---
@app.post("/analyze")
async def analyze_content(
    video: UploadFile = File(None),
    audio: UploadFile = File(None), # For Round 1, this should be a text file (transcript)
    text: UploadFile = File(None),
    source_id: str = Form(None), # NEW: Accept source_id as a form field from frontend
    db: Session = Depends(get_db) # NEW: Inject database session
):
    print(f"DEBUG: === ANALYZE ENDPOINT CALLED ===", file=sys.stderr)
    print(f"DEBUG: video received: {video.filename if video else 'None'}", file=sys.stderr)
    print(f"DEBUG: audio received: {audio.filename if audio else 'None'}", file=sys.stderr)
    print(f"DEBUG: text received: {text.filename if text else 'None'}", file=sys.stderr)
    print(f"DEBUG: source_id received: {source_id if source_id else 'None'}", file=sys.stderr)
    
    temp_files = {} # Dictionary to store paths of saved temporary files

    try:
        # Save uploaded files to temporary locations using the helper
        if video and video.filename: # Check if file object exists and has a filename
            print(f"DEBUG: Processing video file: {video.filename}", file=sys.stderr)
            temp_files["video"] = save_uploaded_file_helper(video, UPLOAD_DIR)
            print(f"DEBUG: Video saved to {temp_files['video']}", file=sys.stderr)
        
        if audio and audio.filename: # Check if file object exists and has a filename
            print(f"DEBUG: Processing audio file: {audio.filename}", file=sys.stderr)
            temp_files["audio"] = save_uploaded_file_helper(audio, UPLOAD_DIR)
            print(f"DEBUG: Audio saved to {temp_files['audio']}", file=sys.stderr)
        
        if text and text.filename: # Check if file object exists and has a filename
            print(f"DEBUG: Processing text file: {text.filename}", file=sys.stderr)
            temp_files["text"] = save_uploaded_file_helper(text, UPLOAD_DIR)
            print(f"DEBUG: Text saved to {temp_files['text']}", file=sys.stderr)

        if not temp_files:
            raise HTTPException(status_code=400, detail="No files provided for analysis. Please upload at least one file.")

        print(f"DEBUG: Files to analyze: {list(temp_files.keys())}", file=sys.stderr)

        # --- 1. Call CM-SDD ---
        print("DEBUG: Calling CM-SDD module...", file=sys.stderr)
        try:
            cm_sdd_results = await analyze_cross_modal(
                video_path=temp_files.get("video"),
                audio_path=temp_files.get("audio"),
                text_path=temp_files.get("text")
            )
            print("DEBUG: CM-SDD module completed successfully.", file=sys.stderr)
        except Exception as e:
            print(f"ERROR: CM-SDD module failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            # Create a fallback result
            cm_sdd_results = {
                "text_content": "",
                "audio_transcript": f"CM-SDD processing failed: {str(e)}",
                "text_sentiment": {"label": "ERROR", "score": 0.0},
                "audio_sentiment_implied_by_transcript": {"label": "ERROR", "score": 0.0},
                "semantic_similarity_scores": {"text_audio": "Error"},
                "discrepancy_detected": True,
                "discrepancy_reason": [f"CM-SDD processing error: {str(e)}"],
                "video_status": f"Processing failed: {str(e)}"
            }

        # --- 2. Call LS-ZLF ---
        print("DEBUG: Calling LS-ZLF module...", file=sys.stderr)
        try:
            ls_zlf_results = await analyze_ls_zlf(temp_files)
            print("DEBUG: LS-ZLF module completed successfully.", file=sys.stderr)
        except Exception as e:
            print(f"ERROR: LS-ZLF module failed: {e}", file=sys.stderr)
            ls_zlf_results = {
                "deepfake_analysis": {
                    "deepfake_detected": False,
                    "reason": f"LS-ZLF processing failed: {str(e)}"
                },
                "llm_origin_analysis": {
                    "llm_origin": "Error",
                    "confidence": 0,
                    "reason": f"LS-ZLF processing failed: {str(e)}"
                }
            }

        # --- 3. Call DP-CNG ---
        print("DEBUG: Calling DP-CNG module...", file=sys.stderr)
        try:
            dp_cng_suggestion = await generate_counter_narrative(
                {"cm_sdd": cm_sdd_results, "ls_zlf": ls_zlf_results, "original_text_path": temp_files.get("text")}
            )
            print("DEBUG: DP-CNG module completed successfully.", file=sys.stderr)
        except Exception as e:
            print(f"ERROR: DP-CNG module failed: {e}", file=sys.stderr)
            dp_cng_suggestion = f"Counter-narrative generation failed: {str(e)}"

        # --- Combine all AI analysis results ---
        full_analysis_results = {
            "cm_sdd": cm_sdd_results,
            "ls_zlf": ls_zlf_results,
            "dp_cng_suggestion": dp_cng_suggestion,
            "overall_status": "All analysis modules completed."
        }

        # --- Source Tracking and Flagging Logic ---
        # Initialize source_status, which will be added to the final response
        source_status = {"source_id": source_id, "flag_count": 0, "is_high_risk": False, "status": "Not Tracked", "risk_level_text": "N/A"}
        
        if source_id: # Only proceed if a source_id was provided in the request
            source_status["status"] = "Tracking"
            
            # Determine if the *current content* should be flagged based on AI analysis results.
            # If CM-SDD detected a discrepancy OR LS-ZLF detected a deepfake (conceptual)
            # OR LS-ZLF detected an AI origin for text.
            is_content_flagged_by_ai = (
                full_analysis_results["cm_sdd"]["discrepancy_detected"] or
                (full_analysis_results["ls_zlf"]["deepfake_analysis"]["deepfake_detected"]) or # Placeholder for deepfake
                (full_analysis_results["ls_zlf"]["llm_origin_analysis"]["llm_origin"] not in ["N/A", "Human/Uncertain"])
            )

            # Query the database for the source_id.
            source_record = db.query(Source).filter(Source.source_id == source_id).first()

            if not source_record:
                # If source_id is new, create a new record in the database
                print(f"DEBUG: Source '{source_id}' not found, creating new record.", flush=True)
                source_record = Source(source_id=source_id, flag_count=0, is_high_risk=False)
                db.add(source_record)
                db.commit() # Commit the transaction to save it to the database
                db.refresh(source_record) # Refresh to load any auto-generated fields (like ID and created_at)

            if is_content_flagged_by_ai:
                # If the content was flagged by our AI modules, increment the source's flag_count
                source_record.flag_count += 1
                
                # --- Define Nuanced Risk Thresholds ---
                # These thresholds define the different risk levels based on flag_count
                MEDIUM_RISK_THRESHOLD = 1 # E.g., 1 flag
                HIGH_RISK_THRESHOLD = 3 # E.g., 3 flags
                CRITICAL_RISK_THRESHOLD = 5 # E.g., 5 flags or more

                # --- Determine is_high_risk based on new thresholds ---
                # 'is_high_risk' is primarily a boolean flag. We'll set it to True if it reaches
                # either HIGH_RISK or CRITICAL_RISK thresholds.
                if source_record.flag_count >= HIGH_RISK_THRESHOLD: # If 3 or more flags
                    source_record.is_high_risk = True
                else: # If less than HIGH_RISK_THRESHOLD (and content was just flagged)
                    source_record.is_high_risk = False 

                # Update the last_flagged_at timestamp
                source_record.last_flagged_at = func.now() 
                
                db.commit() # Commit the changes to the existing record
                db.refresh(source_record) # Refresh to load the updated flag_count, is_high_risk

                source_status["status"] = "Flagged in this analysis"
                print(f"DEBUG: Source '{source_id}' flagged. New count: {source_record.flag_count}.", flush=True)
            else:
                source_status["status"] = "Content not flagged by AI" # Content was clean this time
                print(f"DEBUG: Source '{source_id}' content was clean. Count: {source_record.flag_count}.", flush=True)

            # --- Map flag_count to human-readable risk level text ---
            risk_level_text = "N/A" # Default before checking
            if source_record.flag_count >= CRITICAL_RISK_THRESHOLD:
                risk_level_text = "Critical Risk"
            elif source_record.flag_count >= HIGH_RISK_THRESHOLD:
                risk_level_text = "High Risk"
            elif source_record.flag_count >= MEDIUM_RISK_THRESHOLD:
                risk_level_text = "Medium Risk"
            else: # If flag_count is 0
                risk_level_text = "Low Risk"

            # Update the source_status dictionary to be sent in the response
            source_status["flag_count"] = source_record.flag_count
            source_status["is_high_risk"] = source_record.is_high_risk # Keep this for internal logic
            source_status["risk_level_text"] = risk_level_text # NEW: Add the human-readable text
            
        else: # This block handles the case where no source_id was provided
            print("DEBUG: No source_id provided. Skipping source tracking.", flush=True)

        full_analysis_results["source_tracking"] = source_status # Add tracking info to final response

        print(f"DEBUG: === ANALYSIS COMPLETED SUCCESSFULLY ===", file=sys.stderr)
        return full_analysis_results

    except HTTPException:
        raise # Re-raise HTTPExceptions as they are handled by FastAPI
    except Exception as e:
        # Catch any other unexpected errors during processing
        print(f"ERROR: Unhandled exception during file analysis: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr) # Print full traceback for debugging
        # Re-raise as a 500 error to the client
        raise HTTPException(status_code=500, detail=f"Internal Server Error during analysis: {str(e)}")
    finally:
        # Clean up temporary files
        print(f"DEBUG: Cleaning up {len(temp_files)} temporary files", file=sys.stderr)
        for file_type, path in temp_files.items():
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"DEBUG: Cleaned up temporary file: {path}", file=sys.stderr)
                except OSError as e:
                    print(f"ERROR: Failed to clean up file {path}: {e}", file=sys.stderr)
        print(f"DEBUG: === ANALYZE ENDPOINT FINISHED ===", file=sys.stderr)