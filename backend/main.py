import os
import shutil
import uuid
import sys 

from fastapi import FastAPI, UploadFile, File, HTTPException, Form 
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Depends 

# --- Import your AI modules ---
from .modules.cm_sdd import analyze_cross_modal
from .modules.ls_zlf import analyze_ls_zlf
from .modules.dp_cng import generate_counter_narrative

# --- Database Imports ---
from sqlalchemy.orm import Session 
from sqlalchemy.sql import func 
from .db.database import Base, engine, SessionLocal, get_db 
from .db.models import Source 

# NEW: Import logging and configure it
import logging
from backend.logging_config import configure_logging # Import your logging configuration file
logger = logging.getLogger(__name__) # Use __name__ to get module-specific logger

# --- Configuration ---
UPLOAD_DIR = "uploaded_media"
os.makedirs(UPLOAD_DIR, exist_ok=True) 

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Vigilant Echo Backend",
    description="Autonomous Adversary-Aware Misinformation Counteraction System API",
    version="0.1.0"
)

# Debug function to check environment (now uses logger)
def debug_environment():
    """Debug function to check all dependencies."""
    logger.debug("=== ENVIRONMENT DEBUG ===") # CHANGED: print to logger.debug
    
    # Check FFmpeg
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
        logger.debug(f"FFmpeg: {'✓ Available' if result.returncode == 0 else '✗ Failed'}") # CHANGED
    except Exception: # Catch any exception, including FileNotFoundError
        logger.debug("FFmpeg: ✗ Not found") # CHANGED
    
    # Check Python libraries
    libraries = ['pydub', 'soundfile', 'torch', 'torchaudio', 'transformers', 'sentence_transformers', 'langdetect', 'sqlalchemy']
    for lib in libraries:
        try:
            __import__(lib)
            logger.debug(f"{lib}: ✓ Available") # CHANGED
        except ImportError:
            logger.debug(f"{lib}: ✗ Missing") # CHANGED
    
    logger.debug("=== END DEBUG ===") # CHANGED

# --- Database Initialization on Startup ---
@app.on_event("startup")
def on_startup_create_db_tables():
    # Configure logging first, so subsequent messages go to the log system
    configure_logging() 
    logger.info("Application startup initiated.") # CHANGED: print to logger.info

    logger.info("Creating database tables...") # CHANGED: print to logger.info
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created/checked successfully.") # CHANGED: print to logger.info
        
        # Optional: Run environment debug on startup. Can be removed for production.
        debug_environment()

    except Exception as e:
        logger.error(f"Failed to create database tables: {e}", exc_info=True) # CHANGED: print to logger.error, added exc_info
        raise RuntimeError("Database initialization failed. Check logs for details.") # Re-raise to halt startup

# --- Static Files Setup ---
try:
    frontend_public_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "public")
    if not os.path.exists(frontend_public_dir):
        logger.warning(f"Frontend public directory not found at '{frontend_public_dir}'. Frontend might not load correctly.") # CHANGED: print to logger.warning
    app.mount("/static", StaticFiles(directory=frontend_public_dir), name="static")
except RuntimeError as e:
    logger.error(f"Error mounting static files: {e}. Frontend might not load correctly.", exc_info=True) # CHANGED: print to logger.error, added exc_info


# --- Root Endpoint (Serves Frontend HTML) ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_file_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "public", "index.html")
    
    if not os.path.exists(html_file_path):
        logger.error(f"Frontend HTML file not found at '{html_file_path}'.") # CHANGED: print to logger.error
        raise HTTPException(status_code=404, detail=f"Frontend HTML file not found at '{html_file_path}'. Make sure frontend/public/index.html exists.")
    
    with open(html_file_path, "r", encoding="utf-8") as f:
        logger.debug(f"Serving index.html from {html_file_path}") # CHANGED: print to logger.debug
        return f.read()

# --- Helper function to save uploaded files ---
def save_uploaded_file_helper(uploaded_file: UploadFile, upload_dir: str) -> str:
    logger.debug(f"Attempting to save uploaded file: {uploaded_file.filename}") # CHANGED
    
    if not uploaded_file.filename:
        logger.error("Uploaded file has no filename. Raising ValueError.") # CHANGED
        raise ValueError("Uploaded file has no filename") 
    
    file_extension = os.path.splitext(uploaded_file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(upload_dir, unique_filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(uploaded_file.file, buffer)
        
        # Verify file was saved
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            logger.debug(f"File saved successfully: {file_path} (size: {file_size} bytes)") # CHANGED
            return file_path
        else:
            raise Exception("File was not saved properly (exists check failed).")
            
    except Exception as e:
        logger.error(f"Failed to save file {uploaded_file.filename}: {e}", exc_info=True) # CHANGED
        raise 

# --- NEW: Sources Endpoint (Get All Tracked Sources) ---
# backend/main.py (Find and replace your existing get_tracked_sources function)

# --- NEW: Sources Endpoint (Get All Tracked Sources) ---
@app.get("/sources")
async def get_tracked_sources(db: Session = Depends(get_db)):
    logger.info("GET /sources endpoint called.")
    try:
        # Query all sources from the database, ordered by flag_count descending (most flagged first)
        sources = db.query(Source).order_by(Source.flag_count.desc()).all()

        # Define the thresholds within this function's scope
        MEDIUM_RISK_THRESHOLD = 1 
        HIGH_RISK_THRESHOLD = 3 
        CRITICAL_RISK_THRESHOLD = 5 

        sources_list = []
        for source in sources:
            # Calculate risk_level_text for EACH source here
            current_risk_level_text = "N/A" # Initialize for each source
            if source.flag_count >= CRITICAL_RISK_THRESHOLD:
                current_risk_level_text = "Critical Risk"
            elif source.flag_count >= HIGH_RISK_THRESHOLD:
                current_risk_level_text = "High Risk"
            elif source.flag_count >= MEDIUM_RISK_THRESHOLD:
                current_risk_level_text = "Medium Risk"
            else: # If flag_count is 0
                current_risk_level_text = "Low Risk"

            # Handle last_flagged_at for display (only show if flags > 0)
            last_flagged_display = None
            if source.flag_count > 0 and source.last_flagged_at:
                last_flagged_display = source.last_flagged_at.isoformat()

            sources_list.append({
                "source_id": source.source_id,
                "flag_count": source.flag_count,
                "is_high_risk": source.is_high_risk,
                "created_at": source.created_at.isoformat() if source.created_at else None,
                "last_flagged_at": last_flagged_display,
                "risk_level_text": current_risk_level_text # Add the calculated risk level
            })

        logger.debug(f"Retrieved {len(sources_list)} sources from database.")
        return sources_list

    except Exception as e:
        logger.error(f"Failed to retrieve sources: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error while fetching sources: {str(e)}")

# ... (rest of main.py - analyze_content, save_uploaded_file_helper, etc.) ...
# --- Analysis Endpoint (Receives Multi-Modal Files and Source ID) ---
@app.post("/analyze")
async def analyze_content(
    video: UploadFile = File(None),
    audio: UploadFile = File(None), 
    text: UploadFile = File(None),
    source_id: str = Form(None), 
    db: Session = Depends(get_db)
):
    logger.info("=== ANALYZE ENDPOINT CALLED ===") # CHANGED
    logger.debug(f"Video received: {video.filename if video else 'None'}") # CHANGED
    logger.debug(f"Audio received: {audio.filename if audio else 'None'}") # CHANGED
    logger.debug(f"Text received: {text.filename if text else 'None'}") # CHANGED
    logger.debug(f"Source_id received: {source_id if source_id else 'None'}") # CHANGED
    
    temp_files = {} 

    try:
        # Save uploaded files to temporary locations using the helper
        if video and video.filename: 
            logger.debug(f"Processing video file: {video.filename}") # CHANGED
            temp_files["video"] = save_uploaded_file_helper(video, UPLOAD_DIR)
            logger.debug(f"Video saved to {temp_files['video']}") # CHANGED
        
        if audio and audio.filename: 
            logger.debug(f"Processing audio file: {audio.filename}") # CHANGED
            temp_files["audio"] = save_uploaded_file_helper(audio, UPLOAD_DIR)
            logger.debug(f"Audio saved to {temp_files['audio']}") # CHANGED
        
        if text and text.filename: 
            logger.debug(f"Processing text file: {text.filename}") # CHANGED
            temp_files["text"] = save_uploaded_file_helper(text, UPLOAD_DIR)
            logger.debug(f"Text saved to {temp_files['text']}") # CHANGED

        if not temp_files:
            logger.warning("No files provided for analysis. Raising 400 error.") # CHANGED
            raise HTTPException(status_code=400, detail="No files provided for analysis. Please upload at least one file.")

        logger.debug(f"Files to analyze: {list(temp_files.keys())}") # CHANGED

        # --- 1. Call CM-SDD ---
        logger.debug("Calling CM-SDD module...") # CHANGED
        try:
            cm_sdd_results = await analyze_cross_modal(
                video_path=temp_files.get("video"),
                audio_path=temp_files.get("audio"),
                text_path=temp_files.get("text")
            )
            logger.debug("CM-SDD module completed successfully.") # CHANGED
        except Exception as e:
            logger.error(f"CM-SDD module failed: {e}", exc_info=True) # CHANGED
            cm_sdd_results = { # Fallback result to avoid crashing downstream modules
                "text_content": "",
                "audio_transcript": f"CM-SDD processing failed: {str(e)}",
                "text_sentiment": {"label": "ERROR", "score": 0.0},
                "audio_sentiment_implied_by_transcript": {"label": "ERROR", "score": 0.0},
                "semantic_similarity_scores": {"text_audio": "Error"},
                "discrepancy_detected": True, # Assume discrepancy if module fails
                "discrepancy_reason": [f"CM-SDD processing error: {str(e)}"],
                "video_status": f"Processing failed: {str(e)}",
                "detected_languages": {"text_language": "Error", "audio_language": "Error"} # Fallback for new fields
            }

        # --- 2. Call LS-ZLF ---
        logger.debug("Calling LS-ZLF module...") # CHANGED
        try:
            ls_zlf_results = await analyze_ls_zlf(temp_files)
            logger.debug("LS-ZLF module completed successfully.") # CHANGED
        except Exception as e:
            logger.error(f"LS-ZLF module failed: {e}", exc_info=True) # CHANGED
            ls_zlf_results = { # Fallback result
                "deepfake_analysis": {
                    "deepfake_detected": False, # Cannot detect if module fails
                    "reason": f"LS-ZLF processing failed: {str(e)}"
                },
                "llm_origin_analysis": {
                    "llm_origin": "Error",
                    "confidence": 0,
                    "reason": f"LS-ZLF processing failed: {str(e)}"
                }
            }

        # --- 3. Call DP-CNG ---
        logger.debug("Calling DP-CNG module...") # CHANGED
        try:
            dp_cng_suggestion = await generate_counter_narrative(
                {"cm_sdd": cm_sdd_results, "ls_zlf": ls_zlf_results, "original_text_path": temp_files.get("text")}
            )
            logger.debug("DP-CNG module completed successfully.") # CHANGED
        except Exception as e:
            logger.error(f"DP-CNG module failed: {e}", exc_info=True) # CHANGED
            dp_cng_suggestion = f"Counter-narrative generation failed: {str(e)}"

        # --- Combine all AI analysis results before source tracking ---
        full_analysis_results = {
            "cm_sdd": cm_sdd_results,
            "ls_zlf": ls_zlf_results,
            "dp_cng_suggestion": dp_cng_suggestion,
            "overall_status": "All analysis modules completed."
        }
        
        # --- NEW: Propagation Velocity Estimator (PVE) Logic ---
        # Initialize propagation_velocity
        propagation_velocity = "Slow" # Default if no strong indicators

        # Determine if the content should be considered "High Impact" or "Rapid Spread"
        is_high_impact = False

        # Rule 1: High discrepancy with emotional content (from CM-SDD)
        cm_sdd_discrepancy_detected = cm_sdd_results.get("discrepancy_detected", False)
        cm_sdd_reasons = cm_sdd_results.get("discrepancy_reason", [])
        
        emotional_keywords = ["sentiment mismatch", "emotional", "strong disagreement", "contradiction"]
        if cm_sdd_discrepancy_detected and any(keyword in r.lower() for r in cm_sdd_reasons for keyword in emotional_keywords):
            is_high_impact = True

        # Rule 2: Deepfake detected (from LS-ZLF)
        if ls_zlf_results.get("deepfake_analysis", {}).get("deepfake_detected", False):
            is_high_impact = True
        
        # Rule 3: LLM origin and content is flagged for discrepancy
        llm_origin = ls_zlf_results.get("llm_origin_analysis", {}).get("llm_origin", "N/A")
        if llm_origin not in ["N/A", "Human/Uncertain", "Error"] and cm_sdd_discrepancy_detected:
            is_high_impact = True
            
        # Determine propagation velocity based on high impact flags
        if is_high_impact:
            propagation_velocity = "Rapid"
        elif cm_sdd_discrepancy_detected or (llm_origin not in ["N/A", "Human/Uncertain", "Error"]):
            # If just a discrepancy or LLM origin, but not "high impact"
            propagation_velocity = "Moderate"
        else:
            propagation_velocity = "Slow" # If content seems clean or low impact

        full_analysis_results["propagation_velocity"] = propagation_velocity
        print(f"DEBUG: Propagation Velocity estimated as: {propagation_velocity}", file=sys.stderr)

        # --- Source Tracking and Flagging Logic ---
        source_status = {"source_id": source_id, "flag_count": 0, "is_high_risk": False, "status": "Not Tracked", "risk_level_text": "N/A"}
        
        if source_id: 
            source_status["status"] = "Tracking"
            
            is_content_flagged_by_ai = (
                full_analysis_results["cm_sdd"]["discrepancy_detected"] or
                (full_analysis_results["ls_zlf"]["deepfake_analysis"]["deepfake_detected"]) or 
                (full_analysis_results["ls_zlf"]["llm_origin_analysis"]["llm_origin"] not in ["N/A", "Human/Uncertain"])
            )

            source_record = db.query(Source).filter(Source.source_id == source_id).first()

            if not source_record:
                logger.debug(f"Source '{source_id}' not found, creating new record.") # CHANGED
                source_record = Source(source_id=source_id, flag_count=0, is_high_risk=False)
                db.add(source_record)
                db.commit() 
                db.refresh(source_record) 

            if is_content_flagged_by_ai:
                source_record.flag_count += 1
                
                MEDIUM_RISK_THRESHOLD = 1 
                HIGH_RISK_THRESHOLD = 3 
                CRITICAL_RISK_THRESHOLD = 5 

                if source_record.flag_count >= HIGH_RISK_THRESHOLD: 
                    source_record.is_high_risk = True
                else: 
                    source_record.is_high_risk = False 

                source_record.last_flagged_at = func.now() 
                
                db.commit() 
                db.refresh(source_record) 

                source_status["status"] = "Flagged in this analysis"
                logger.debug(f"Source '{source_id}' flagged. New count: {source_record.flag_count}.") # CHANGED
            else:
                source_status["status"] = "Content not flagged by AI" 
                logger.debug(f"Source '{source_id}' content was clean. Count: {source_record.flag_count}.") # CHANGED

            risk_level_text = "N/A" 
            if source_record.flag_count >= CRITICAL_RISK_THRESHOLD:
                risk_level_text = "Critical Risk"
            elif source_record.flag_count >= HIGH_RISK_THRESHOLD:
                risk_level_text = "High Risk"
            elif source_record.flag_count >= MEDIUM_RISK_THRESHOLD:
                risk_level_text = "Medium Risk"
            else: 
                risk_level_text = "Low Risk"

            source_status["flag_count"] = source_record.flag_count
            source_status["is_high_risk"] = source_record.is_high_risk 
            source_status["risk_level_text"] = risk_level_text 
            
        else: 
            logger.debug("No source_id provided. Skipping source tracking.") # CHANGED

        full_analysis_results["source_tracking"] = source_status 

        logger.info("=== ANALYSIS COMPLETED SUCCESSFULLY ===") # CHANGED
        return full_analysis_results

    except HTTPException:
        raise 
    except Exception as e:
        logger.error(f"Unhandled exception during file analysis: {e}", exc_info=True) # CHANGED
        raise HTTPException(status_code=500, detail=f"Internal Server Error during analysis: {str(e)}")
    finally:
        logger.debug(f"Cleaning up {len(temp_files)} temporary files.") # CHANGED
        for file_type, path in temp_files.items():
            if os.path.exists(path):
                try:
                    os.remove(path)
                    logger.debug(f"Cleaned up temporary file: {path}.") # CHANGED
                except OSError as e:
                    logger.error(f"Failed to clean up file {path}: {e}", exc_info=True) # CHANGED
        logger.info(f"=== ANALYZE ENDPOINT FINISHED ===") # CHANGED