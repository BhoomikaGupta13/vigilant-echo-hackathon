import torch
import os
import io
import numpy as np
import cv2
from PIL import Image
import sys
import uuid
import subprocess

# Check for FFmpeg installation first
def check_ffmpeg_installation():
    """Check if FFmpeg is installed and accessible."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("DEBUG: FFmpeg is installed and accessible", file=sys.stderr)
            return True
        else:
            print("DEBUG: FFmpeg command failed", file=sys.stderr)
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
        print(f"DEBUG: FFmpeg not found or not accessible: {e}", file=sys.stderr)
        return False

# Check FFmpeg availability
FFMPEG_AVAILABLE = check_ffmpeg_installation()

try:
    from pydub import AudioSegment
    from pydub.utils import which
    print("DEBUG: pydub imported successfully", file=sys.stderr)
    
    # Check what pydub can find
    ffmpeg_path = which("ffmpeg")
    print(f"DEBUG: pydub found ffmpeg at: {ffmpeg_path}", file=sys.stderr)
    
except ImportError as e:
    print(f"ERROR: Could not import pydub: {e}", file=sys.stderr)
    AudioSegment = None

try:
    import torchaudio
    print("DEBUG: torchaudio imported successfully", file=sys.stderr)
except ImportError as e:
    print(f"ERROR: Could not import torchaudio: {e}", file=sys.stderr)
    torchaudio = None

try:
    import soundfile as sf
    print("DEBUG: soundfile imported successfully", file=sys.stderr)
except ImportError as e:
    print(f"ERROR: Could not import soundfile: {e}", file=sys.stderr)
    sf = None

try:
    import librosa
    print("DEBUG: librosa imported successfully", file=sys.stderr)
except ImportError as e:
    print(f"ERROR: Could not import librosa: {e}", file=sys.stderr)
    librosa = None

# --- Hugging Face Transformers ---
try:
    from transformers import pipeline, Wav2Vec2Processor, Wav2Vec2ForCTC
    print("DEBUG: transformers imported successfully", file=sys.stderr)
except ImportError as e:
    print(f"ERROR: Could not import transformers: {e}", file=sys.stderr)

# --- Sentence Transformers for semantic similarity ---
try:
    from sentence_transformers import SentenceTransformer, util
    print("DEBUG: sentence_transformers imported successfully", file=sys.stderr)
except ImportError as e:
    print(f"ERROR: Could not import sentence_transformers: {e}", file=sys.stderr)

# --- Load pre-trained AI models ---

# 1. For text sentiment analysis
print("DEBUG: Loading text sentiment analyzer model...", file=sys.stderr)
try:
    text_sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    print("DEBUG: Text sentiment analyzer loaded successfully.", file=sys.stderr)
except Exception as e:
    print(f"ERROR: Failed to load text sentiment analyzer: {e}", file=sys.stderr)
    text_sentiment_analyzer = None

# 2. For semantic embeddings
print("DEBUG: Loading sentence embedder model...", file=sys.stderr)
try:
    sentence_embedder = SentenceTransformer('all-MiniLM-L6-v2')
    print("DEBUG: Sentence embedder loaded successfully.", file=sys.stderr)
except Exception as e:
    print(f"ERROR: Failed to load sentence embedder: {e}", file=sys.stderr)
    sentence_embedder = None

# 3. For Audio Transcription (Wav2Vec2)
print("DEBUG: Loading Wav2Vec2 for audio transcription...", file=sys.stderr)
try:
    speech_to_text_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    speech_to_text_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    print("DEBUG: Wav2Vec2 loaded successfully.", file=sys.stderr)
except Exception as e:
    print(f"ERROR: Failed to load Wav2Vec2: {e}. Audio transcription will be disabled.", file=sys.stderr)
    speech_to_text_processor = None
    speech_to_text_model = None

# --- Helper functions ---

def extract_text_sentiment(text_content):
    """Extract sentiment from text content."""
    print(f"DEBUG: extract_text_sentiment called with content length: {len(text_content) if text_content else 0}", file=sys.stderr)
    
    if text_sentiment_analyzer is None:
        return {'label': 'ERROR', 'score': 0.0, 'detail': 'Sentiment model not loaded.'}
    if not text_content or not text_content.strip():
        return {'label': 'NEUTRAL', 'score': 1.0, 'detail': 'No text content provided.'}
    try:
        result = text_sentiment_analyzer(text_content, truncation=True, max_length=512)[0]
        print(f"DEBUG: Sentiment analysis result: {result}", file=sys.stderr)
        return {'label': result['label'], 'score': float(result['score'])}
    except Exception as e:
        print(f"ERROR: Sentiment analysis failed: {e}", file=sys.stderr)
        return {'label': 'ERROR', 'score': 0.0, 'detail': str(e)}

def extract_audio_from_video_simple(video_path, output_dir):
    """Extract audio using FFmpeg directly (more reliable than pydub for some formats)."""
    if not FFMPEG_AVAILABLE:
        print("ERROR: FFmpeg not available for audio extraction", file=sys.stderr)
        return None
        
    try:
        print(f"DEBUG: Extracting audio from video using FFmpeg: {video_path}", file=sys.stderr)
        
        # Create unique temporary file path
        temp_audio_path = os.path.join(output_dir, f"temp_audio_{uuid.uuid4()}.wav")
        
        # Use FFmpeg directly to extract audio
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # 16-bit PCM
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite output file
            temp_audio_path
        ]
        
        print(f"DEBUG: Running FFmpeg command: {' '.join(cmd)}", file=sys.stderr)
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            if os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 0:
                print(f"DEBUG: Audio extracted successfully to: {temp_audio_path}", file=sys.stderr)
                return temp_audio_path
            else:
                print("ERROR: Audio file was created but is empty", file=sys.stderr)
                return None
        else:
            print(f"ERROR: FFmpeg failed: {result.stderr}", file=sys.stderr)
            return None
            
    except subprocess.TimeoutExpired:
        print("ERROR: FFmpeg timed out during audio extraction", file=sys.stderr)
        return None
    except Exception as e:
        print(f"ERROR: Failed to extract audio from video: {e}", file=sys.stderr)
        return None

def extract_audio_from_video_pydub(video_path, output_dir):
    """Extract audio using pydub (fallback method)."""
    if AudioSegment is None:
        print("ERROR: pydub not available", file=sys.stderr)
        return None
        
    try:
        print(f"DEBUG: Extracting audio from video using pydub: {video_path}", file=sys.stderr)
        
        # Load video and extract audio
        audio = AudioSegment.from_file(video_path)
        
        # Create unique temporary file path
        temp_audio_path = os.path.join(output_dir, f"temp_audio_{uuid.uuid4()}.wav")
        
        # Convert to mono and set sample rate
        audio = audio.set_channels(1)  # Convert to mono
        audio = audio.set_frame_rate(16000)  # Set to 16kHz for Wav2Vec2
        audio.export(temp_audio_path, format="wav")
        
        if os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 0:
            print(f"DEBUG: Audio extracted successfully to: {temp_audio_path}", file=sys.stderr)
            return temp_audio_path
        else:
            print("ERROR: Audio file was created but is empty", file=sys.stderr)
            return None
            
    except Exception as e:
        print(f"ERROR: Failed to extract audio using pydub: {e}", file=sys.stderr)
        return None

def transcribe_audio_from_file(audio_file_path):
    """Transcribe audio from a given file path using Wav2Vec2."""
    print(f"DEBUG: transcribe_audio_from_file called with: {audio_file_path}", file=sys.stderr)
    
    if speech_to_text_model is None or speech_to_text_processor is None:
        error_msg = "Audio transcription model not loaded."
        print(f"ERROR: {error_msg}", file=sys.stderr)
        return error_msg
    
    if not audio_file_path or not os.path.exists(audio_file_path):
        error_msg = f"Audio file not found: {audio_file_path}"
        print(f"ERROR: {error_msg}", file=sys.stderr)
        return "No audio file provided for transcription."
    
    # Check file size
    file_size = os.path.getsize(audio_file_path)
    print(f"DEBUG: Audio file size: {file_size} bytes", file=sys.stderr)
    
    if file_size == 0:
        return "Audio file is empty."
    
    try:
        print(f"DEBUG: Starting transcription of {audio_file_path}", file=sys.stderr)
        
        # Try librosa first (most reliable)
        if librosa is not None:
            try:
                print("DEBUG: Attempting to load audio with librosa", file=sys.stderr)
                # Load with librosa (automatically handles resampling)
                speech, sampling_rate = librosa.load(audio_file_path, sr=16000, mono=True, duration=30)  # Limit to 30s for demo
                print(f"DEBUG: Audio loaded with librosa. Shape: {speech.shape}, SR: {sampling_rate}", file=sys.stderr)
            except Exception as librosa_error:
                print(f"DEBUG: Librosa failed: {librosa_error}", file=sys.stderr)
                speech = None
        else:
            speech = None
        
        # Fallback to soundfile
        if speech is None and sf is not None:
            try:
                print("DEBUG: Attempting to load audio with soundfile", file=sys.stderr)
                speech, sampling_rate = sf.read(audio_file_path)
                
                # Handle stereo
                if speech.ndim > 1:
                    speech = speech.mean(axis=1)
                
                # Resample if needed
                if sampling_rate != 16000:
                    if librosa is not None:
                        speech = librosa.resample(speech, orig_sr=sampling_rate, target_sr=16000)
                    else:
                        print("WARNING: Cannot resample audio without librosa", file=sys.stderr)
                
                # Limit duration for demo (30 seconds max)
                max_samples = 16000 * 30  # 30 seconds at 16kHz
                if len(speech) > max_samples:
                    speech = speech[:max_samples]
                    
                print(f"DEBUG: Audio loaded with soundfile. Shape: {speech.shape}", file=sys.stderr)
            except Exception as sf_error:
                print(f"ERROR: Soundfile also failed: {sf_error}", file=sys.stderr)
                return f"Could not load audio file: {sf_error}"
        
        if speech is None:
            return "Could not load audio file with available libraries."
        
        # Ensure speech is a 1D numpy array
        speech = np.array(speech, dtype=np.float32)
        print(f"DEBUG: Audio preprocessed. Shape: {speech.shape}, dtype: {speech.dtype}", file=sys.stderr)
        
        # Check for silence
        if np.max(np.abs(speech)) < 0.01:
            return "Audio appears to be silent or very quiet."
        
        # Process with Wav2Vec2
        print("DEBUG: Processing with Wav2Vec2...", file=sys.stderr)
        inputs = speech_to_text_processor(
            speech, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        )
        
        print("DEBUG: Audio processed through Wav2Vec2 processor", file=sys.stderr)
        
        # Perform inference
        with torch.no_grad():
            logits = speech_to_text_model(inputs.input_values).logits
        
        # Get predicted token IDs
        predicted_ids = torch.argmax(logits, dim=-1)
        
        # Decode to text
        transcript = speech_to_text_processor.batch_decode(predicted_ids)[0]
        
        print(f"DEBUG: Raw transcription: '{transcript}'", file=sys.stderr)
        
        # Clean up the transcript
        transcript = transcript.strip().lower()
        
        if not transcript:
            return "No speech detected in audio."
        
        print(f"DEBUG: Final transcription: '{transcript}'", file=sys.stderr)
        return transcript
        
    except torch.OutOfMemoryError as e:
        error_msg = f"Out of memory during transcription: {str(e)}"
        print(f"ERROR: {error_msg}", file=sys.stderr)
        return error_msg
    except Exception as e:
        error_msg = f"Audio transcription failed: {str(e)}"
        print(f"ERROR: {error_msg}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return error_msg

# --- Main analysis function ---

async def analyze_cross_modal(video_path: str = None, audio_path: str = None, text_path: str = None):
    """Main cross-modal analysis function with comprehensive debugging."""
    print("DEBUG: === Starting analyze_cross_modal ===", file=sys.stderr)
    print(f"DEBUG: video_path: {video_path}", file=sys.stderr)
    print(f"DEBUG: audio_path: {audio_path}", file=sys.stderr)
    print(f"DEBUG: text_path: {text_path}", file=sys.stderr)
    
    results = {
        "text_content": "",
        "audio_transcript": "",
        "text_sentiment": {},
        "audio_sentiment_implied_by_transcript": {},
        "semantic_similarity_scores": {},
        "discrepancy_detected": False,
        "discrepancy_reason": [],
        "video_status": "No video file provided."
    }

    # 1. Process Main Text Input
    print("DEBUG: Processing text input...", file=sys.stderr)
    if text_path and os.path.exists(text_path):
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                text_content = f.read().strip()
                results["text_content"] = text_content
                print(f"DEBUG: Text content loaded: {len(text_content)} characters", file=sys.stderr)
                if text_content:
                    results["text_sentiment"] = extract_text_sentiment(text_content)
                else:
                    results["text_sentiment"] = {'label': 'NEUTRAL', 'score': 1.0, 'detail': 'Empty text file.'}
        except Exception as e:
            print(f"ERROR: Could not read text file {text_path}: {e}", file=sys.stderr)
            results["text_content"] = "Error reading text."
            results["text_sentiment"] = {'label': 'ERROR', 'score': 0.0, 'detail': str(e)}
    else:
        print("DEBUG: No text file provided", file=sys.stderr)
        results["text_sentiment"] = {'label': 'NEUTRAL', 'score': 1.0, 'detail': 'No text file provided.'}

    # 2. Handle Audio Processing
    print("DEBUG: Processing audio/video input...", file=sys.stderr)
    audio_source_path_for_asr = None
    temp_audio_file_path = None

    if video_path and os.path.exists(video_path):
        print(f"DEBUG: Video file exists: {video_path}", file=sys.stderr)
        print(f"DEBUG: Video file size: {os.path.getsize(video_path)} bytes", file=sys.stderr)
        
        results["video_status"] = "Video file provided. Attempting audio extraction."
        
        # Try to extract audio from video
        upload_dir = os.path.dirname(video_path) if os.path.dirname(video_path) else "."
        
        # Try FFmpeg first, then pydub
        temp_audio_file_path = extract_audio_from_video_simple(video_path, upload_dir)
        if not temp_audio_file_path:
            print("DEBUG: FFmpeg extraction failed, trying pydub...", file=sys.stderr)
            temp_audio_file_path = extract_audio_from_video_pydub(video_path, upload_dir)
        
        if temp_audio_file_path:
            audio_source_path_for_asr = temp_audio_file_path
            results["video_status"] = "Audio extracted from video successfully."
            print(f"DEBUG: Audio extraction successful: {temp_audio_file_path}", file=sys.stderr)
        else:
            results["video_status"] = "Failed to extract audio from video."
            print("ERROR: All audio extraction methods failed", file=sys.stderr)
            
    elif audio_path and os.path.exists(audio_path):
        print(f"DEBUG: Standalone audio file provided: {audio_path}", file=sys.stderr)
        results["video_status"] = "Standalone audio file provided."
        audio_source_path_for_asr = audio_path
    else:
        print("DEBUG: No video or audio file provided", file=sys.stderr)
        results["video_status"] = "No video or audio file provided for transcription."

    # 3. Transcribe Audio
    print("DEBUG: Starting audio transcription...", file=sys.stderr)
    if audio_source_path_for_asr:
        print(f"DEBUG: Transcribing audio from: {audio_source_path_for_asr}", file=sys.stderr)
        results["audio_transcript"] = transcribe_audio_from_file(audio_source_path_for_asr)
        
        # Analyze sentiment of the transcript
        if (results["audio_transcript"] and 
            not results["audio_transcript"].startswith("Audio transcription") and
            not results["audio_transcript"].startswith("No audio") and
            not results["audio_transcript"].startswith("Could not")):
            results["audio_sentiment_implied_by_transcript"] = extract_text_sentiment(results["audio_transcript"])
        else:
            results["audio_sentiment_implied_by_transcript"] = {'label': 'N/A', 'score': 0.0, 'detail': 'No valid transcript.'}
    else:
        print("DEBUG: No audio source available for transcription", file=sys.stderr)
        results["audio_transcript"] = "No audio file provided for transcription."
        results["audio_sentiment_implied_by_transcript"] = {'label': 'N/A', 'score': 0.0, 'detail': 'No audio to transcribe.'}
    
    # 4. Semantic Similarity & Discrepancy Detection
    print("DEBUG: Performing semantic similarity analysis...", file=sys.stderr)
    if sentence_embedder is None:
        results["discrepancy_reason"].append("Semantic embedder model not loaded, cannot perform similarity check.")
    else:
        # Check if we have valid content from both sources
        text_valid = results["text_content"] and results["text_content"].strip()
        audio_valid = (results["audio_transcript"] and 
                      results["audio_transcript"].strip() and 
                      not results["audio_transcript"].startswith("Audio transcription") and
                      not results["audio_transcript"].startswith("No audio") and
                      not results["audio_transcript"].startswith("Could not") and
                      results["audio_transcript"] != "No speech detected in audio.")
        
        print(f"DEBUG: Text valid: {text_valid}, Audio valid: {audio_valid}", file=sys.stderr)
        
        if text_valid and audio_valid:
            try:
                embeddings = sentence_embedder.encode(
                    [results["text_content"], results["audio_transcript"]],
                    convert_to_tensor=True
                )
                similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
                results["semantic_similarity_scores"]["text_audio"] = similarity
                print(f"DEBUG: Semantic similarity: {similarity:.3f}", file=sys.stderr)

                # Discrepancy threshold
                SIMILARITY_THRESHOLD = 0.7
                if similarity < SIMILARITY_THRESHOLD:
                    results["discrepancy_detected"] = True
                    results["discrepancy_reason"].append(
                        f"Low semantic similarity between text and audio transcript ({similarity:.2f} < {SIMILARITY_THRESHOLD})."
                    )
                    
            except Exception as e:
                print(f"ERROR: Semantic similarity calculation failed: {e}", file=sys.stderr)
                results["semantic_similarity_scores"]["text_audio"] = "Error"
                results["discrepancy_reason"].append(f"Semantic similarity calculation failed: {str(e)}")
        else:
            results["semantic_similarity_scores"]["text_audio"] = "N/A"
            missing_items = []
            if not text_valid:
                missing_items.append("valid text content")
            if not audio_valid:
                missing_items.append("valid audio transcript")
            results["discrepancy_reason"].append(f"Missing {' and '.join(missing_items)} for semantic comparison.")

    # 5. Sentiment Mismatch Detection
    print("DEBUG: Performing sentiment mismatch analysis...", file=sys.stderr)
    text_label = results["text_sentiment"].get('label')
    audio_label = results["audio_sentiment_implied_by_transcript"].get('label')
    
    print(f"DEBUG: Text sentiment: {text_label}, Audio sentiment: {audio_label}", file=sys.stderr)

    if (text_label and audio_label and 
        text_label not in ['ERROR', 'N/A'] and 
        audio_label not in ['ERROR', 'N/A']):
        
        if text_label != audio_label:
            results["discrepancy_detected"] = True
            results["discrepancy_reason"].append(
                f"Sentiment mismatch: Text is '{text_label}' while audio transcript is '{audio_label}'."
            )
    else:
        results["discrepancy_reason"].append("Could not perform sentiment comparison (missing valid sentiment data).")

    # Clean up temporary files
    if temp_audio_file_path and os.path.exists(temp_audio_file_path):
        try:
            os.remove(temp_audio_file_path)
            print(f"DEBUG: Cleaned up temporary audio file: {temp_audio_file_path}", file=sys.stderr)
        except OSError as e:
            print(f"ERROR: Failed to clean up temp audio file {temp_audio_file_path}: {e}", file=sys.stderr)

    # Final status
    if not results["discrepancy_reason"]:
        results["discrepancy_reason"].append("No discrepancies detected.")
        results["discrepancy_detected"] = False

    print(f"DEBUG: CM-SDD analysis completed. Discrepancy detected: {results['discrepancy_detected']}", file=sys.stderr)
    print(f"DEBUG: Final results summary: {len(results['discrepancy_reason'])} reasons", file=sys.stderr)
    print("DEBUG: === End analyze_cross_modal ===", file=sys.stderr)
    
    return results