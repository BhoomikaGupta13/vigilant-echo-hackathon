# backend/modules/multilingual_processor.py

from langdetect import detect, DetectorFactory, lang_detect_exception
import sys

# Set seed for reproducibility (important for langdetect sometimes)
DetectorFactory.seed = 0

def detect_language(text_content: str) -> str:
    """
    Detects the language of a given text content.
    Returns 'unknown' or 'error' if detection fails or text is too short.
    """
    if not text_content or not text_content.strip():
        return "N/A (no content)"

    # langdetect needs a minimum amount of text
    if len(text_content.strip()) < 5: 
        return "N/A (too short)"

    try:
        # detect() returns the language code (e.g., 'en', 'hi', 'fr')
        lang_code = detect(text_content)
        return lang_code
    except lang_detect_exception.LangDetectException as e:
        # This exception is raised if it cannot detect (e.g., very short text, non-text)
        print(f"DEBUG: Language detection failed: {e} for text: '{text_content[:50]}...'", file=sys.stderr)
        return "unknown"
    except Exception as e:
        # Catch other unexpected errors
        print(f"ERROR: Unexpected error during language detection: {e}", file=sys.stderr)
        return "error"