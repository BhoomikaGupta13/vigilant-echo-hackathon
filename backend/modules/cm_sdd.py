import torch
import os
import io
import numpy as np
import cv2
from PIL import Image
import sys

# --- Hugging Face Transformers ---
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq, AutoModel

# --- Sentence Transformers for semantic similarity ---
from sentence_transformers import SentenceTransformer, util

# --- NLTK for text (already downloaded stopwords and punkt) ---
# You might not strictly use NLTK here, but it's often good for text prep.

# --- Load pre-trained models (these will download on first run) ---
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

# 3. For (optional) audio transcription - HIGHLY OPTIONAL FOR 6-HOUR SPRINT
# Only uncomment and implement if you have spare time AND small audio files for demo.
# Otherwise, rely on text input for 'audio_transcript'.
# print("Loading Wav2Vec2 for audio transcription (optional)...", file=sys.stderr)
# try:
#     speech_to_text_processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
#     speech_to_text_model = AutoModelForSpeechSeq2Seq.from_pretrained("facebook/wav2vec2-base-960h")
#     print("Wav2Vec2 loaded.", file=sys.stderr)
# except Exception as e:
#     print(f"Error loading Wav2Vec2: {e}. Audio transcription will be skipped.", file=sys.stderr)
#     speech_to_text_processor = None
#     speech_to_text_model = None

# 4. For (optional) video feature extraction - VERY SIMPLIFIED/SKIP FOR 6-HOUR SPRINT
# This model is quite large. For 6 hours, it's best to skip actual video feature extraction
# and use placeholders or only focus on text/audio discrepancies.
# print("Loading image feature extractor (for video, optional)...", file=sys.stderr)
# try:
#     image_feature_extractor = pipeline("feature-extraction", model="facebook/detr-resnet-50", framework="pt")
#     print("Image feature extractor loaded.", file=sys.stderr)
# except Exception as e:
#     print(f"Error loading image feature extractor: {e}. Video analysis will be skipped.", file=sys.stderr)
#     image_feature_extractor = None


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

def transcribe_audio_hackathon_simplified(audio_path):
    """
    HACKATHON SIMPLIFICATION:
    For Round 1 (6-hour sprint), we assume the 'audio_path' actually points
    to a text file that contains the audio's transcript.
    This avoids complex audio processing/large model downloads during the sprint.
    """
    if audio_path and os.path.exists(audio_path):
        try:
            with open(audio_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    return content.lower()
                else:
                    return "no audio content detected"
        except Exception as e:
            print(f"Error reading audio (as text) file {audio_path}: {e}", file=sys.stderr)
            return "audio transcription failed"
    return "no audio file provided"

# Full audio transcription (uncomment if you have time/resources)
# def transcribe_audio_full(audio_path):
#     if speech_to_text_model is None or speech_to_text_processor is None:
#         return "Audio transcription model not loaded."
#     if not audio_path or not os.path.exists(audio_path):
#         return "No audio file provided."
#     try:
#         import soundfile as sf
#         speech, sampling_rate = sf.read(audio_path)
#         if sampling_rate != 16000:
#             import torchaudio
#             waveform = torch.from_numpy(speech).float()
#             if waveform.dim() == 2: waveform = waveform.mean(dim=1) # handle stereo
#             resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
#             speech = resampler(waveform).numpy()
#         
#         input_values = speech_to_text_processor(speech, sampling_rate=16000, return_tensors="pt").input_values
#         logits = speech_to_text_model(input_values).logits
#         predicted_ids = torch.argmax(logits, dim=-1)
#         transcript = speech_to_text_processor.batch_decode(predicted_ids)[0]
#         return transcript.lower()
#     except Exception as e:
#         print(f"Error transcribing audio {audio_path}: {e}", file=sys.stderr)
#         return f"audio transcription failed: {str(e)}"

# Video feature extraction (uncomment if you have time/resources, but likely too slow for 6h)
# def extract_video_features(video_path, max_frames=1): # Limit to 1 frame for speed
#     if image_feature_extractor is None: return np.array([])
#     cap = cv2.VideoCapture(video_path)
#     ret, frame = cap.read()
#     cap.release()
#     if not ret: return np.array([])
#     
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     pil_image = Image.fromarray(rgb_frame)
#     try:
#         image_embeddings = image_feature_extractor(images=pil_image, return_tensors="pt")[0]
#         return image_embeddings.mean(dim=0).flatten().numpy() # Simplified
#     except Exception as e:
#         print(f"Error extracting video frame features: {e}", file=sys.stderr)
#         return np.array([])


# --- Main analysis function ---

async def analyze_cross_modal(video_path: str = None, audio_path: str = None, text_path: str = None):
    results = {
        "text_content": "",
        "audio_transcript": "",
        "text_sentiment": {},
        "audio_sentiment_implied_by_transcript": {},
        "semantic_similarity_scores": {},
        "discrepancy_detected": False,
        "discrepancy_reason": []
    }

    # 1. Process Main Text Input
    if text_path and os.path.exists(text_path):
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
                results["text_content"] = text_content
                results["text_sentiment"] = extract_text_sentiment(text_content)
        except Exception as e:
            print(f"Error reading main text file {text_path}: {e}", file=sys.stderr)
            results["text_content"] = "Error reading text."

    # 2. Process Audio (using hackathon simplification: treating audio file as its transcript)
    # For a full implementation, you'd use transcribe_audio_full(audio_path)
    audio_transcript_content = transcribe_audio_hackathon_simplified(audio_path)
    results["audio_transcript"] = audio_transcript_content
    results["audio_sentiment_implied_by_transcript"] = extract_text_sentiment(audio_transcript_content)

    # 3. Process Video (conceptual/placeholder for 6-hour sprint)
    # For a hackathon, we might not extract features directly due to time/compute.
    # Just acknowledge its presence or absence.
    if video_path and os.path.exists(video_path):
        results["video_status"] = "Video file provided (features not extracted for 6-hour demo)."
        # If you uncomment extract_video_features, you'd process it here
        # video_features = extract_video_features(video_path)
        # if video_features.size > 0:
        #     results["video_features_status"] = "Features extracted."
        # else:
        #     results["video_features_status"] = "Feature extraction failed."
    else:
        results["video_status"] = "No video file provided."

    # 4. Semantic Similarity & Discrepancy Detection (Text vs. Audio Transcript)
    if sentence_embedder is None:
        results["discrepancy_reason"].append("Semantic embedder model not loaded, cannot perform similarity check.")
        return results # Exit early if model not loaded

    if results["text_content"] and results["audio_transcript"] and \
       results["text_content"].strip() and results["audio_transcript"].strip():

        # Ensure text is not just placeholders from errors
        if "Error" in results["text_content"] or "failed" in results["audio_transcript"]:
            results["discrepancy_reason"].append("Skipping semantic comparison due to content processing errors.")
            return results

        try:
            embeddings = sentence_embedder.encode(
                [results["text_content"], results["audio_transcript"]],
                convert_to_tensor=True
            )
            similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
            results["semantic_similarity_scores"]["text_audio"] = similarity

            # Threshold for semantic discrepancy (adjust as needed for demo)
            if similarity < 0.7: # Example threshold
                results["discrepancy_detected"] = True
                results["discrepancy_reason"].append(
                    f"Low semantic similarity between main text and audio transcript ({similarity:.2f} < 0.70)."
                )
        except Exception as e:
            print(f"Error during semantic similarity calculation: {e}", file=sys.stderr)
            results["discrepancy_reason"].append(f"Semantic similarity calculation failed: {str(e)}")
    else:
        results["semantic_similarity_scores"]["text_audio"] = "N/A"
        results["discrepancy_reason"].append("Not enough text content in both modalities for semantic comparison.")


    # 5. Sentiment Mismatch Detection (Text vs. Audio Transcript)
    text_label = results["text_sentiment"].get('label')
    audio_label = results["audio_sentiment_implied_by_transcript"].get('label')

    if text_label and audio_label and text_label != 'ERROR' and audio_label != 'ERROR':
        if text_label != audio_label:
            results["discrepancy_detected"] = True
            results["discrepancy_reason"].append(
                f"Sentiment mismatch: Main text is '{text_label}' "
                f"while audio transcript is inferred as '{audio_label}'."
            )
        elif text_label == 'NEUTRAL' and results["text_sentiment"].get('score', 0) < 0.6 and \
             audio_label == 'NEUTRAL' and results["audio_sentiment_implied_by_transcript"].get('score', 0) < 0.6:
            # If both are neutral but with low confidence, could indicate ambiguity/issue
            results["discrepancy_reason"].append(
                "Both text and audio transcript sentiments are neutral, but with low confidence."
            )
    else:
        results["discrepancy_reason"].append("Sentiment analysis could not be performed for all modalities.")

    # Ensure discrepancy_detected is True if any reason was added
    if results["discrepancy_reason"] and not results["discrepancy_detected"]:
        results["discrepancy_detected"] = True # Force true if a reason exists

    return results