import os
import shutil
import uuid
import sys
import logging # Ensure logging is imported

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Depends

# --- Import your AI modules ---
from .modules.cm_sdd import analyze_cross_modal
from .modules.ls_zlf import analyze_ls_zlf
from .modules.dp_cng import generate_counter_narrative
from .modules.visual_deepfake_analyzer import analyze_visual_deepfake # NEW: Import the visual deepfake analyzer
# --- Database Imports ---
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from .db.database import Base, engine, SessionLocal, get_db
from .db.models import Source

# NEW: Import logging and configure it
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
    logger.debug("=== ENVIRONMENT DEBUG ===")
    
    # Check FFmpeg
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
        logger.debug(f"FFmpeg: {'✓ Available' if result.returncode == 0 else '✗ Failed'}")
    except Exception:
        logger.debug("FFmpeg: ✗ Not found")
    
    # Check Python libraries
    # Added new dependencies for visual analysis
    libraries = [
        'pydub', 'soundfile', 'torch', 'torchaudio', 'transformers',
        'sentence_transformers', 'langdetect', 'sqlalchemy', 'cv2',
        'PIL', 'skimage', 'matplotlib', 'dlib', 'imutils', 'scipy',
        'mediapipe', 'efficientnet_pytorch', 'pytorch_grad_cam'
    ]
    for lib in libraries:
        try:
            # Handle specific import names for packages
            if lib == 'cv2': __import__('cv2')
            elif lib == 'PIL': __import__('PIL')
            elif lib == 'skimage': __import__('skimage')
            elif lib == 'matplotlib': __import__('matplotlib')
            elif lib == 'dlib': __import__('dlib')
            elif lib == 'imutils': __import__('imutils')
            elif lib == 'scipy': __import__('scipy')
            elif lib == 'efficientnet_pytorch': __import__('efficientnet_pytorch')
            elif lib == 'pytorch_grad_cam': __import__('pytorch_grad_cam')
            else: __import__(lib)
            logger.debug(f"{lib}: ✓ Available")
        except ImportError:
            logger.debug(f"{lib}: ✗ Missing")
        except Exception as e: # Catch other potential initialization errors
            logger.debug(f"{lib}: ✗ Error during import/init: {e}")
    
    logger.debug("=== END DEBUG ===")

# --- Database Initialization on Startup ---
@app.on_event("startup")
def on_startup_create_db_tables():
    configure_logging()
    logger.info("Application startup initiated.")

    logger.info("Creating database tables...")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created/checked successfully.")
        
        debug_environment() # Run environment debug on startup

    except Exception as e:
        logger.error(f"Failed to create database tables: {e}", exc_info=True)
        raise RuntimeError("Database initialization failed. Check logs for details.")

# --- Static Files Setup ---
try:
    frontend_public_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "public")
    if not os.path.exists(frontend_public_dir):
        logger.warning(f"Frontend public directory not found at '{frontend_public_dir}'. Frontend might not load correctly.")
    app.mount("/static", StaticFiles(directory=frontend_public_dir), name="static")
except RuntimeError as e:
    logger.error(f"Error mounting static files: {e}. Frontend might not load correctly.", exc_info=True)


# --- Root Endpoint (Serves Frontend HTML) ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_file_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "public", "index.html")
    
    if not os.path.exists(html_file_path):
        logger.error(f"Frontend HTML file not found at '{html_file_path}'.")
        raise HTTPException(status_code=404, detail=f"Frontend HTML file not found at '{html_file_path}'. Make sure frontend/public/index.html exists.")
    
    with open(html_file_path, "r", encoding="utf-8") as f:
        logger.debug(f"Serving index.html from {html_file_path}")
        return f.read()

# --- Helper function to save uploaded files ---
def save_uploaded_file_helper(uploaded_file: UploadFile, upload_dir: str) -> str:
    logger.debug(f"Attempting to save uploaded file: {uploaded_file.filename}")
    
    if not uploaded_file.filename:
        logger.error("Uploaded file has no filename. Raising ValueError.")
        raise ValueError("Uploaded file has no filename")
    
    file_extension = os.path.splitext(uploaded_file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(upload_dir, unique_filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(uploaded_file.file, buffer)
        
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            logger.debug(f"File saved successfully: {file_path} (size: {file_size} bytes)")
            return file_path
        else:
            raise Exception("File was not saved properly (exists check failed).")
            
    except Exception as e:
        logger.error(f"Failed to save file {uploaded_file.filename}: {e}", exc_info=True)
        raise

# --- Sources Endpoint (Get All Tracked Sources) ---
@app.get("/sources")
async def get_tracked_sources(db: Session = Depends(get_db)):
    logger.info("GET /sources endpoint called.")
    try:
        sources = db.query(Source).order_by(Source.flag_count.desc()).all()

        # Define the thresholds within this function's scope
        MEDIUM_RISK_THRESHOLD = 2 
        HIGH_RISK_THRESHOLD = 3 
        CRITICAL_RISK_THRESHOLD = 5 

        sources_list = []
        for source in sources:
            current_risk_level_text = "N/A"
            if source.flag_count >= CRITICAL_RISK_THRESHOLD:
                current_risk_level_text = "Critical Risk"
            elif source.flag_count >= HIGH_RISK_THRESHOLD:
                current_risk_level_text = "High Risk"
            elif source.flag_count >= MEDIUM_RISK_THRESHOLD:
                current_risk_level_text = "Medium Risk"
            else:
                current_risk_level_text = "Low Risk"

            last_flagged_display = None
            if source.flag_count > 0 and source.last_flagged_at:
                last_flagged_display = source.last_flagged_at.isoformat()

            sources_list.append({
                "source_id": source.source_id,
                "flag_count": source.flag_count,
                "is_high_risk": source.is_high_risk,
                "created_at": source.created_at.isoformat() if source.created_at else None,
                "last_flagged_at": last_flagged_display,
                "risk_level_text": current_risk_level_text
            })

        logger.debug(f"Retrieved {len(sources_list)} sources from database.")
        return sources_list

    except Exception as e:
        logger.error(f"Failed to retrieve sources: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error while fetching sources: {str(e)}")

# --- Analysis Endpoint (Receives Multi-Modal Files and Source ID) ---
@app.post("/analyze")
async def analyze_content(
    video: UploadFile = File(None),
    audio: UploadFile = File(None),
    text: UploadFile = File(None),
    source_id: str = Form(None),
    db: Session = Depends(get_db)
):
    logger.info("=== ANALYZE ENDPOINT CALLED ===")
    logger.debug(f"Video received: {video.filename if video else 'None'}")
    logger.debug(f"Audio received: {audio.filename if audio else 'None'}")
    logger.debug(f"Text received: {text.filename if text else 'None'}")
    logger.debug(f"Source_id received: {source_id if source_id else 'None'}")
    
    temp_files = {}
    analysis_results = {} # Initialize dictionary to hold all analysis results

    try:
        # Save uploaded files to temporary locations
        if video and video.filename:
            logger.debug(f"Processing video file: {video.filename}")
            temp_files["video"] = save_uploaded_file_helper(video, UPLOAD_DIR)
            logger.debug(f"Video saved to {temp_files['video']}")
            
            # --- Enhanced Visual Deepfake Analysis for video ---
            logger.debug("Calling Enhanced Visual Deepfake Analyzer module for video...")
            try:
                visual_analysis = await analyze_visual_deepfake(temp_files["video"])
                analysis_results["visual_analysis"] = visual_analysis
                logger.debug("Enhanced Visual Deepfake Analyzer completed successfully.")
                
                # Calculate overall deepfake likelihood from visual analysis
                if visual_analysis and len(visual_analysis) > 0:
                    deepfake_probs = [face.get("deepfake_probability", 0.0) for face in visual_analysis if "deepfake_probability" in face]
                    if deepfake_probs:
                        avg_deepfake_prob = sum(deepfake_probs) / len(deepfake_probs)
                        max_deepfake_prob = max(deepfake_probs)
                        
                        # Use weighted average (70% max, 30% average) for overall assessment
                        overall_prob = (max_deepfake_prob * 0.7) + (avg_deepfake_prob * 0.3)
                        
                        if overall_prob > 0.7:
                            analysis_results["overall_deepfake_likelihood"] = f"HIGH FAKE PROBABILITY ({overall_prob*100:.1f}%)"
                        elif overall_prob > 0.5:
                            analysis_results["overall_deepfake_likelihood"] = f"MODERATE FAKE PROBABILITY ({overall_prob*100:.1f}%)"
                        elif overall_prob > 0.3:
                            analysis_results["overall_deepfake_likelihood"] = f"LOW FAKE PROBABILITY ({overall_prob*100:.1f}%)"
                        else:
                            analysis_results["overall_deepfake_likelihood"] = f"LIKELY REAL ({(1-overall_prob)*100:.1f}% authentic)"
                    else:
                        analysis_results["overall_deepfake_likelihood"] = "N/A (No face probability data)"
                else:
                    analysis_results["overall_deepfake_likelihood"] = "N/A (No faces detected)"
                    
            except Exception as e:
                logger.error(f"Enhanced Visual Deepfake Analyzer failed: {e}", exc_info=True)
                analysis_results["visual_analysis"] = []
                analysis_results["overall_deepfake_likelihood"] = f"Visual analysis failed: {str(e)}"

        if audio and audio.filename:
            logger.debug(f"Processing audio file: {audio.filename}")
            temp_files["audio"] = save_uploaded_file_helper(audio, UPLOAD_DIR)
            logger.debug(f"Audio saved to {temp_files['audio']}")
        
        if text and text.filename:
            logger.debug(f"Processing text file: {text.filename}")
            temp_files["text"] = save_uploaded_file_helper(text, UPLOAD_DIR)
            logger.debug(f"Text saved to {temp_files['text']}")

        if not temp_files:
            logger.warning("No files provided for analysis. Raising 400 error.")
            raise HTTPException(status_code=400, detail="No files provided for analysis. Please upload at least one file.")

        logger.debug(f"Files to analyze: {list(temp_files.keys())}")

        # --- Call CM-SDD (if audio or text is present) ---
        if temp_files.get("audio") or temp_files.get("text") or temp_files.get("video"):
            # CM-SDD can extract audio from video, so if video is present, it can run.
            logger.debug("Calling CM-SDD module...")
            try:
                cm_sdd_results = await analyze_cross_modal(
                    video_path=temp_files.get("video"),
                    audio_path=temp_files.get("audio"),
                    text_path=temp_files.get("text")
                )
                analysis_results["cm_sdd"] = cm_sdd_results
                logger.debug("CM-SDD module completed successfully.")
            except Exception as e:
                logger.error(f"CM-SDD module failed: {e}", exc_info=True)
                analysis_results["cm_sdd"] = {
                    "text_content": "", "audio_transcript": f"CM-SDD processing failed: {str(e)}",
                    "text_sentiment": {"label": "ERROR", "score": 0.0},
                    "audio_sentiment_implied_by_transcript": {"label": "ERROR", "score": 0.0},
                    "semantic_similarity_scores": {"text_audio": "Error"},
                    "discrepancy_detected": True, "discrepancy_reason": [f"CM-SDD processing error: {str(e)}"],
                    "video_status": f"Processing failed: {str(e)}", "detected_languages": {}
                }
        else:
            logger.debug("Skipping CM-SDD as no audio, text, or video input provided.")
            analysis_results["cm_sdd"] = {
                "text_content": "", "audio_transcript": "", "text_sentiment": {},
                "audio_sentiment_implied_by_transcript": {}, "semantic_similarity_scores": {},
                "discrepancy_detected": False, "discrepancy_reason": ["No audio, text, or video provided for CM-SDD."],
                "video_status": "Not processed by CM-SDD", "detected_languages": {}
            }

        # --- Call LS-ZLF (if text or audio transcript is present) ---
        # LS-ZLF typically works on text/linguistic data.
        # It needs direct text input or a transcript generated by CM-SDD.
        if temp_files.get("text") or analysis_results.get("cm_sdd", {}).get("audio_transcript"):
            logger.debug("Calling LS-ZLF module...")
            try:
                ls_zlf_results = await analyze_ls_zlf(temp_files)
                analysis_results["ls_zlf"] = ls_zlf_results
                logger.debug("LS-ZLF module completed successfully.")
            except Exception as e:
                logger.error(f"LS-ZLF module failed: {e}", exc_info=True)
                analysis_results["ls_zlf"] = {
                    "deepfake_analysis": {"deepfake_detected": False, "reason": f"LS-ZLF processing failed: {str(e)}"},
                    "llm_origin_analysis": {"llm_origin": "Error", "confidence": 0, "reason": f"LS-ZLF processing failed: {str(e)}"}
                }
        else:
            logger.debug("Skipping LS-ZLF as no text or audio transcript is available.")
            analysis_results["ls_zlf"] = {
                "deepfake_analysis": {"deepfake_detected": False, "reason": "No text or audio transcript for LS-ZLF."},
                "llm_origin_analysis": {"llm_origin": "N/A", "confidence": 0, "reason": "No text or audio transcript for LS-ZLF."}
            }

        # --- Call DP-CNG (if CM-SDD or LS-ZLF results are present and meaningful) ---
        if analysis_results.get("cm_sdd") or analysis_results.get("ls_zlf"):
            logger.debug("Calling DP-CNG module...")
            try:
                dp_cng_suggestion = await generate_counter_narrative(
                    {"cm_sdd": analysis_results.get("cm_sdd", {}), "ls_zlf": analysis_results.get("ls_zlf", {}), "original_text_path": temp_files.get("text")}
                )
                analysis_results["dp_cng_suggestion"] = dp_cng_suggestion
                logger.debug("DP-CNG module completed successfully.")
            except Exception as e:
                logger.error(f"DP-CNG module failed: {e}", exc_info=True)
                analysis_results["dp_cng_suggestion"] = f"Counter-narrative generation failed: {str(e)}"
        else:
            logger.debug("Skipping DP-CNG as no relevant analysis results are available.")
            analysis_results["dp_cng_suggestion"] = "No counter-narrative generated due to missing analysis results."

        # --- Propagation Velocity Estimator (PVE) Logic (Enhanced with Visual Analysis) ---
        propagation_velocity = "Slow"
        is_high_impact = False

        cm_sdd_results = analysis_results.get("cm_sdd", {})
        cm_sdd_discrepancy_detected = cm_sdd_results.get("discrepancy_detected", False)
        cm_sdd_reasons = cm_sdd_results.get("discrepancy_reason", [])
        
        emotional_keywords = ["sentiment mismatch", "emotional", "strong disagreement", "contradiction"]
        if cm_sdd_discrepancy_detected and any(keyword in r.lower() for r in cm_sdd_reasons for keyword in emotional_keywords):
            is_high_impact = True

        ls_zlf_results = analysis_results.get("ls_zlf", {})
        visual_analysis_results = analysis_results.get("visual_analysis", [])

        # Rule 2: Enhanced Deepfake detected (from LS-ZLF) OR Visual Deepfake detected
        visual_high_fake = False
        if visual_analysis_results:
            high_fake_faces = [face for face in visual_analysis_results if face.get("deepfake_probability", 0.0) > 0.7]
            if high_fake_faces:
                visual_high_fake = True
                
        if (ls_zlf_results.get("deepfake_analysis", {}).get("deepfake_detected", False) or visual_high_fake):
            is_high_impact = True
        
        # Rule 3: LLM origin and content is flagged for discrepancy
        llm_origin = ls_zlf_results.get("llm_origin_analysis", {}).get("llm_origin", "N/A")
        if llm_origin not in ["N/A", "Human/Uncertain", "Error"] and cm_sdd_discrepancy_detected:
            is_high_impact = True
            
        # Rule 4: Multiple visual anomalies detected
        if visual_analysis_results:
            anomaly_count = sum([
                face.get("blink_anomaly_detected", False) + 
                face.get("micro_expression_anomaly_detected", False) + 
                face.get("corneal_reflection_anomaly_detected", False)
                for face in visual_analysis_results
            ])
            if anomaly_count >= 2:
                is_high_impact = True
            
        if is_high_impact:
            propagation_velocity = "Rapid"
        elif (cm_sdd_discrepancy_detected or 
              (llm_origin not in ["N/A", "Human/Uncertain", "Error"]) or 
              (visual_analysis_results and any(face.get("deepfake_probability", 0.0) > 0.5 for face in visual_analysis_results))):
            propagation_velocity = "Moderate"
        else:
            propagation_velocity = "Slow"

        analysis_results["propagation_velocity"] = propagation_velocity
        logger.debug(f"Propagation Velocity estimated as: {propagation_velocity}")

        # --- Enhanced Source Tracking and Flagging Logic ---
        source_status = {"source_id": source_id, "flag_count": 0, "is_high_risk": False, "status": "Not Tracked", "risk_level_text": "N/A"}
        
        if source_id:
            source_status["status"] = "Tracking"
            
            # Enhanced flagging logic to include comprehensive visual analysis
            visual_flagged = False
            if visual_analysis_results:
                # Flag if any face has high deepfake probability or multiple anomalies
                for face in visual_analysis_results:
                    if (face.get("deepfake_probability", 0.0) > 0.7 or 
                        sum([face.get("blink_anomaly_detected", False), 
                             face.get("micro_expression_anomaly_detected", False), 
                             face.get("corneal_reflection_anomaly_detected", False)]) >= 2):
                        visual_flagged = True
                        break
            
            is_content_flagged_by_ai = (
                analysis_results.get("cm_sdd", {}).get("discrepancy_detected", False) or
                analysis_results.get("ls_zlf", {}).get("deepfake_analysis", {}).get("deepfake_detected", False) or
                (analysis_results.get("ls_zlf", {}).get("llm_origin_analysis", {}).get("llm_origin") not in ["N/A", "Human/Uncertain"]) or
                visual_flagged  # Enhanced visual flagging
            )

            source_record = db.query(Source).filter(Source.source_id == source_id).first()

            if not source_record:
                logger.debug(f"Source '{source_id}' not found, creating new record.")
                source_record = Source(source_id=source_id, flag_count=0, is_high_risk=False)
                db.add(source_record)
                db.commit()
                db.refresh(source_record)

            if is_content_flagged_by_ai:
                source_record.flag_count += 1
                
                MEDIUM_RISK_THRESHOLD = 2
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
                logger.debug(f"Source '{source_id}' flagged. New count: {source_record.flag_count}.")
            else:
                source_status["status"] = "Content not flagged by AI"
                logger.debug(f"Source '{source_id}' content was clean. Count: {source_record.flag_count}.")

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
            logger.debug("No source_id provided. Skipping source tracking.")

        analysis_results["source_tracking"] = source_status

        logger.info("=== ANALYSIS COMPLETED SUCCESSFULLY ===")
        return analysis_results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unhandled exception during file analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error during analysis: {str(e)}")
    finally:
        logger.debug(f"Cleaning up {len(temp_files)} temporary files.")
        for file_type, path in temp_files.items():
            if os.path.exists(path):
                try:
                    os.remove(path)
                    logger.debug(f"Cleaned up temporary file: {path}.")
                except OSError as e:
                    logger.error(f"Failed to clean up file {path}: {e}", exc_info=True)
        logger.info(f"=== ANALYZE ENDPOINT FINISHED ===")