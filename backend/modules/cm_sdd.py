import torch
import os
import io
import numpy as np
import cv2
from PIL import Image
import sys
import uuid # Needed for unique temp filenames for extracted audio
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq, AutoModel

from pydub import AudioSegment # For audio extraction from video
import torchaudio # Required for resampling audio for Wav2Vec2
import soundfile as sf # For reading audio files (used by torchaudio)

# --- Hugging Face Transformers ---
# AutoModelForCTC is for Wav2Vec2, AutoModelForSpeechSeq2Seq is for models like Whisper
from transformers import pipeline, AutoProcessor, AutoModelForCTC, AutoModel

# --- Sentence Transformers for semantic similarity ---
from sentence_transformers import SentenceTransformer, util

# --- Language Detection Library ---
# This is the actual library, NOT a local module import
from langdetect import detect, DetectorFactory, lang_detect_exception 

# Set seed for reproducibility for langdetect (important for consistent results)
DetectorFactory.seed = 0

# --- Load pre-trained AI models (these will download on first run) ---

# 1. For text sentiment analysis (used for both text and audio transcript sentiment)
print("Loading text sentiment analyzer model...", file=sys.stderr)
try:
    text_sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    print("Text sentiment analyzer loaded.", file=sys.stderr)
except Exception as e:
    print(f"Error loading text sentiment analyzer: {e}", file=sys.stderr)
    text_sentiment_analyzer = None # Handle cases where model loading fails

# 2. For semantic embeddings (for text and audio transcript comparison)
print("Loading sentence embedder model...", file=sys.stderr)
try:
    sentence_embedder = SentenceTransformer('all-MiniLM-L6-v2')
    print("Sentence embedder loaded.", file=sys.stderr)
except Exception as e:
    print(f"Error loading sentence embedder: {e}", file=sys.stderr)
    sentence_embedder = None

# 3. For Audio Transcription (Wav2Vec2 - now actively used)
print("Loading Whisper ASR model for audio transcription...", file=sys.stderr)
try:
    # Use openai/whisper-tiny for smallest model (approx 150MB)
    # For better quality (larger size): openai/whisper-base (approx 500MB)
    whisper_model_name = "openai/whisper-tiny" 
    speech_to_text_processor = AutoProcessor.from_pretrained(whisper_model_name)
    # Ensure model is loaded with AutoModelForSpeechSeq2Seq for Whisper
    speech_to_text_model = AutoModelForSpeechSeq2Seq.from_pretrained(whisper_model_name)
    print(f"{whisper_model_name} loaded.", file=sys.stderr)
except Exception as e:
    print(f"Error loading Whisper ASR model: {e}. Audio transcription will be skipped.", file=sys.stderr)
    speech_to_text_processor = None
    speech_to_text_model = None


# --- Helper functions ---

def extract_text_sentiment(text_content):
    if text_sentiment_analyzer is None:
        return {'label': 'ERROR', 'score': 0.0, 'detail': 'Sentiment model not loaded.'}
    if not text_content or not text_content.strip():
        return {'label': 'NEUTRAL', 'score': 1.0, 'detail': 'No text content provided.'}
    try:
        result = text_sentiment_analyzer(text_content, truncation=True, max_length=512)[0]
        return {'label': result['label'], 'score': float(result['score'])}
    except Exception as e:
        print(f"Error in text sentiment analysis: {e}", file=sys.stderr)
        return {'label': 'ERROR', 'score': 0.0, 'detail': str(e)}

# Function to transcribe audio from a given file path (audio or extracted video audio)
# Fully multilingual - supports automatic language detection for any language
def transcribe_audio_from_file(audio_file_path):
    if speech_to_text_model is None or speech_to_text_processor is None:
        return "Audio transcription model not loaded."
    if not audio_file_path or not os.path.exists(audio_file_path):
        return "No audio file provided for transcription."

    try:
        # Read audio file. Specify dtype='float32' to ensure consistency.
        # sf.read returns (data_array, sampling_rate)
        audio_data, sampling_rate = sf.read(audio_file_path, dtype='float32')

        # If audio_data is a scalar (single value, e.g., from a very short/empty segment),
        # convert it to a 1D numpy array to ensure .ndim exists.
        if audio_data.ndim == 0:
            audio_data = np.array([audio_data])
        elif len(audio_data) == 0:
            return "Audio file is empty or corrupted."

        # Check if audio is too short (less than 0.5 seconds)
        min_audio_length = int(0.5 * sampling_rate)  # 0.5 seconds minimum
        if len(audio_data) < min_audio_length:
            print(f"DEBUG: Audio too short ({len(audio_data)} samples), padding...", file=sys.stderr)
            # Pad with silence to meet minimum length
            padding_needed = min_audio_length - len(audio_data)
            audio_data = np.pad(audio_data, (0, padding_needed), mode='constant', constant_values=0)

        # Handle stereo audio by taking the mean across channels to convert to mono (if ndim > 1)
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        # Convert numpy array to torch tensor for resampling
        waveform = torch.from_numpy(audio_data).float()

        # Add batch dimension if not present
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # Resample audio to 16kHz if necessary (Whisper input requirement)
        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Ensure minimum length after resampling (8000 samples = 0.5 seconds at 16kHz)
        min_samples_16k = 8000
        if waveform.shape[1] < min_samples_16k:
            padding_needed = min_samples_16k - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding_needed), mode='constant', value=0)

        # Process with Whisper ASR model
        # input_features expects numpy array, so convert back from tensor
        # Remove batch dimension for processing
        audio_array = waveform.squeeze(0).numpy()
        
        # Process the audio with the processor
        input_features = speech_to_text_processor(
            audio_array, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features

        print(f"DEBUG: Processing audio with automatic multilingual detection...", file=sys.stderr)
        print(f"DEBUG: Input features shape: {input_features.shape}", file=sys.stderr)
        
        # Generate transcription with adjusted parameters for better stability
        with torch.no_grad():
            try:
                # First attempt with conservative parameters
                predicted_ids = speech_to_text_model.generate(
                    input_features,
                    max_new_tokens=50,  # Reduced for stability
                    min_new_tokens=1,   # Ensure at least 1 token is generated
                    num_beams=1,        # Reduced for stability
                    do_sample=False,
                    task="transcribe",
                    use_cache=True,
                    pad_token_id=speech_to_text_processor.tokenizer.pad_token_id,
                    eos_token_id=speech_to_text_processor.tokenizer.eos_token_id
                )
            except Exception as e:
                print(f"DEBUG: First attempt failed: {e}, trying with minimal parameters...", file=sys.stderr)
                # Fallback with minimal parameters
                predicted_ids = speech_to_text_model.generate(
                    input_features,
                    max_length=50,      # Use max_length instead of max_new_tokens
                    num_beams=1,
                    do_sample=False,
                    task="transcribe"
                )

        # Handle the case where predicted_ids might be a single tensor or batch
        if predicted_ids.ndim == 1:
            # Single sequence, add batch dimension
            predicted_ids = predicted_ids.unsqueeze(0)
        
        print(f"DEBUG: Generated tokens shape: {predicted_ids.shape}", file=sys.stderr)
        
        # Decode the predicted IDs to get the text transcript
        # batch_decode expects a batch, so we pass the full tensor
        transcript = speech_to_text_processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]

        # Clean up the transcript but preserve original language formatting
        cleaned_transcript = transcript.strip()
        
        if not cleaned_transcript:
            return "No speech detected."
        
        print(f"DEBUG: Multilingual transcription successful: '{cleaned_transcript[:100]}...'", file=sys.stderr)
        
        # Don't force to lowercase to preserve language-specific formatting
        return cleaned_transcript
        
    except Exception as e:
        print(f"Error during audio transcription from {audio_file_path}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return f"Audio transcription failed: {str(e)}"

# Function to detect language of a given text content - Enhanced for multilingual support
def detect_language(text_content: str) -> str:
    """
    Detects the language of a given text content.
    Enhanced to handle multilingual and mixed-language content.
    Returns language code or descriptive status if detection fails.
    """
    if not text_content or not text_content.strip():
        return "N/A (no content)"
    
    if len(text_content.strip()) < 3:  # Reduced threshold for better detection
        return "N/A (too short)"

    try:
        # Detect primary language
        lang_code = detect(text_content)
        
        # Try to detect if it's mixed language content
        # Split text into chunks and detect language of each
        words = text_content.split()
        if len(words) > 10:  # Only for longer texts
            chunks = [' '.join(words[i:i+5]) for i in range(0, len(words), 5)]
            detected_langs = []
            
            for chunk in chunks[:5]:  # Check first 5 chunks
                try:
                    chunk_lang = detect(chunk)
                    detected_langs.append(chunk_lang)
                except:
                    continue
            
            # If multiple languages detected, indicate mixed content
            unique_langs = list(set(detected_langs))
            if len(unique_langs) > 1:
                return f"{lang_code} (mixed: {', '.join(unique_langs)})"
        
        return lang_code
        
    except lang_detect_exception.LangDetectException as e:
        # For very short or ambiguous text, try a more permissive approach
        try:
            # Use a different detection strategy for short/mixed content
            from langdetect import detect_langs
            probabilities = detect_langs(text_content)
            if probabilities:
                top_lang = probabilities[0]
                if top_lang.prob > 0.7:
                    return top_lang.lang
                else:
                    # Multiple languages detected with low confidence
                    top_langs = [lang.lang for lang in probabilities[:2]]
                    return f"mixed ({', '.join(top_langs)})"
            else:
                return "unknown"
        except:
            print(f"DEBUG: Language detection failed for text: '{text_content[:50]}...'", file=sys.stderr)
            return "unknown"
            
    except Exception as e:
        print(f"ERROR: Unexpected error during language detection: {e}", file=sys.stderr)
        return "error"


# --- Main analysis function ---

async def analyze_cross_modal(video_path: str = None, audio_path: str = None, text_path: str = None):
    results = {
        "text_content": "",
        "audio_transcript": "", # This will now be truly transcribed
        "text_sentiment": {},
        "audio_sentiment_implied_by_transcript": {},
        "semantic_similarity_scores": {},
        "discrepancy_detected": False,
        "discrepancy_reason": [],
        "video_status": "No video file provided.",
        "detected_languages": { # Fields for detected languages
            "text_language": "N/A",
            "audio_language": "N/A"
        }
    }

    # 1. Process Main Text Input
    if text_path and os.path.exists(text_path):
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
                results["text_content"] = text_content
                
                # Detect language for main text
                text_lang = detect_language(text_content)
                results["detected_languages"]["text_language"] = text_lang

                # Perform sentiment analysis for English content or mixed content with English
                if text_lang == 'en' or 'en' in text_lang.lower():
                    results["text_sentiment"] = extract_text_sentiment(text_content)
                else:
                    results["text_sentiment"] = {'label': 'N/A', 'score': 0.0, 'detail': f'Sentiment analysis limited to English content. Detected: {text_lang}'}
        except Exception as e:
            print(f"Error reading main text file {text_path}: {e}", file=sys.stderr)
            results["text_content"] = "Error reading text."
            results["detected_languages"]["text_language"] = "error"


    # 2. Extract Audio from Video (If video provided) OR use standalone audio file
    audio_source_path_for_asr = None
    temp_audio_file_path = None # To store path of extracted audio if from video

    if video_path and os.path.exists(video_path):
        results["video_status"] = "Video file provided. Attempting audio extraction."
        print(f"DEBUG: Extracting audio from video: {video_path}", file=sys.stderr)
        try:
            video_audio = AudioSegment.from_file(video_path)
            temp_audio_file_path = os.path.join(os.path.dirname(video_path), f"temp_audio_{uuid.uuid4()}.wav")
            video_audio.export(temp_audio_file_path, format="wav") # Export audio as WAV
            audio_source_path_for_asr = temp_audio_file_path
            results["video_status"] = "Audio extracted from video."
            print(f"DEBUG: Audio extracted to: {temp_audio_file_path}", file=sys.stderr)
        except Exception as e:
            print(f"ERROR: Error extracting audio from video {video_path}: {e}", file=sys.stderr)
            results["video_status"] = f"Audio extraction from video failed: {str(e)}. (Check FFmpeg installation)"
            audio_source_path_for_asr = None 
    elif audio_path and os.path.exists(audio_path):
        results["video_status"] = "Standalone audio file provided."
        audio_source_path_for_asr = audio_path
    else:
        results["video_status"] = "No video or audio file provided for transcription."


    # 3. Transcribe Audio (from extracted video audio or standalone audio)
    if audio_source_path_for_asr:
        audio_transcript_content = transcribe_audio_from_file(audio_source_path_for_asr)
        results["audio_transcript"] = audio_transcript_content
        
        # Detect language for audio transcript
        audio_lang = detect_language(audio_transcript_content)
        results["detected_languages"]["audio_language"] = audio_lang

        # Perform sentiment analysis for English content or mixed content with English
        if audio_lang == 'en' or 'en' in audio_lang.lower():
            results["audio_sentiment_implied_by_transcript"] = extract_text_sentiment(audio_transcript_content)
        else:
            results["audio_sentiment_implied_by_transcript"] = {'label': 'N/A', 'score': 0.0, 'detail': f'Sentiment analysis limited to English content. Detected: {audio_lang}'}
    else:
        results["audio_transcript"] = "No audio processed."
        results["audio_sentiment_implied_by_transcript"] = {'label': 'N/A', 'score': 0.0, 'detail': 'No audio to transcribe.'}
    
    # 4. Semantic Similarity & Discrepancy Detection (Text vs. Audio Transcript)
    if sentence_embedder is None:
        results["discrepancy_reason"].append("Semantic embedder model not loaded, cannot perform similarity check.")
        return results 

    # Check for valid text content from both sources before attempting semantic comparison
    if results["text_content"] and results["audio_transcript"] and \
       results["text_content"].strip() and results["audio_transcript"].strip() and \
       results["audio_transcript"] != "Audio transcription model not loaded." and \
       not results["audio_transcript"].startswith("Audio transcription failed"): 
        
        try:
            embeddings = sentence_embedder.encode(
                [results["text_content"], results["audio_transcript"]],
                convert_to_tensor=True
            )
            similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
            results["semantic_similarity_scores"]["text_audio"] = similarity

            if similarity < 0.7: 
                results["discrepancy_detected"] = True
                results["discrepancy_reason"].append(
                    f"Low semantic similarity between main text and audio transcript ({similarity:.2f} < 0.70)."
                )
        except Exception as e:
            print(f"Error during semantic similarity calculation: {e}", file=sys.stderr)
            results["discrepancy_reason"].append(f"Semantic similarity calculation failed: {str(e)}")
    else:
        results["semantic_similarity_scores"]["text_audio"] = "N/A"
        results["discrepancy_reason"].append("Not enough valid text content in both modalities for semantic comparison.")


    # 5. Sentiment Mismatch Detection (Text vs. Audio Transcript)
    text_label = results["text_sentiment"].get('label')
    audio_label = results["audio_sentiment_implied_by_transcript"].get('label')

    if text_label and audio_label and text_label != 'ERROR' and audio_label != 'ERROR' and \
       audio_label != 'N/A' and text_label != 'N/A': # Ensure both sentiments were actually processed
        if text_label != audio_label:
            results["discrepancy_detected"] = True
            results["discrepancy_reason"].append(
                f"Sentiment mismatch: Main text is '{text_label}' "
                f"while audio transcript is inferred as '{audio_label}'."
            )
        elif text_label == 'NEUTRAL' and results["text_sentiment"].get('score', 0) < 0.6 and \
             audio_label == 'NEUTRAL' and results["audio_sentiment_implied_by_transcript"].get('score', 0) < 0.6:
            results["discrepancy_reason"].append(
                "Both text and audio transcript sentiments are neutral, but with low confidence."
            )
    else:
        results["discrepancy_reason"].append("Sentiment analysis could not be performed for all modalities (e.g., missing valid English content).")

    # Ensure discrepancy_detected is True if any reason was added
    if results["discrepancy_reason"] and not results["discrepancy_detected"]:
        results["discrepancy_detected"] = True

    # Final cleanup of temporary audio file if it was extracted from video
    if temp_audio_file_path and os.path.exists(temp_audio_file_path):
        try:
            os.remove(temp_audio_file_path)
            print(f"DEBUG: Cleaned up temporary extracted audio file: {temp_audio_file_path}", file=sys.stderr)
        except OSError as e:
            print(f"ERROR: Failed to clean up temp audio file {temp_audio_file_path}: {e}", file=sys.stderr)

    return results