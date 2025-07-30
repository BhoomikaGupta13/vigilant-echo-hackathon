import torch
import torch.nn as nn # For nn.Module
import torchvision.transforms as transforms # For image_transform (for preprocessing PIL images)
from efficientnet_pytorch import EfficientNet # For EfficientNet backbone

import os
import io # For handling image data in memory
import numpy as np
import cv2 # OpenCV for image/video processing
from PIL import Image # Pillow for image manipulation
import sys # For debug prints

# For Grad-CAM (Assumes pytorch-grad-cam is installed or will be)
from pytorch_grad_cam import GradCAM 
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget 
from pytorch_grad_cam.utils.image import show_cam_on_image 

# For MediaPipe Face Detection
import mediapipe as mp 

# For LLM Fingerprinting (existing)
from transformers import pipeline, AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoModel
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import re
import base64 # For encoding image data to send to frontend


# --- GLOBAL VARIABLES FOR MODELS AND DEVICES ---
# These will hold the loaded models and devices so they are loaded only once.
# Note: Text sentiment, sentence embedder, and ASR models are primarily loaded in cm_sdd.py.
# We'll rely on cm_sdd.py to load them and potentially pass their results if needed directly.
# Here, we focus on LS-ZLF specific models.
_visual_deepfake_model_instance = None # For your trained deepfake model
_detection_device = None # For the device your deepfake model is on

_llm_tokenizer = None 
_llm_model = None

_mp_face_detection = None # NEW: Global variable for MediaPipe Face Detection
_mp_drawing = None        # NEW: Global variable for MediaPipe Drawing Utilities


# --- MODEL LOADING FUNCTIONS ---
# Functions to load models (called once at startup for LS-ZLF specific models)

# Your Trained Deepfake Model Class
class EnhancedDeepfakeDetector(nn.Module):
    def __init__(self, model_name='efficientnet-b0'):
        super(EnhancedDeepfakeDetector, self).__init__()
        # Load pretrained EfficientNet model *without* default ImageNet weights initially.
        # This allows us to load YOUR .pth weights cleanly into this instance.
        self.backbone = EfficientNet.from_name(model_name, num_classes=1) # Corrected: use from_name, no weights here
    def forward(self, x):
        # Ensure the output matches what your trained model expects before the final classifier
        # For a binary classifier, often a single output neuron after sigmoid.
        # This assumes your .pth trained model is a full classifier
        return self.backbone(x)

# Function to load your trained deepfake model (called once globally)
def _load_deepfake_model(model_path="efficientnet_b0_deepfake.pth", model_name='efficientnet-b0'): # Make sure model_path is correct filename
    global _visual_deepfake_model_instance, _detection_device
    global _mp_face_detection, _mp_drawing # <-- ADDED: Declare MediaPipe globals

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading deepfake model on device: {device}", file=sys.stderr)
    
    try:
        # Define the absolute path to your trained model .pth file
        current_dir = os.path.dirname(__file__)
        trained_model_path_abs = os.path.abspath(os.path.join(current_dir, "..", "models", model_path))
        
        if not os.path.exists(trained_model_path_abs):
            raise FileNotFoundError(f"Trained deepfake model not found at: {trained_model_path_abs}. Please place your .pth model file in backend/models/ and update the filename if needed.")

        # Instantiate your model class
        model = EnhancedDeepfakeDetector(model_name=model_name).to(device)
        
        # Load the state dictionary from your .pth file
        model.load_state_dict(torch.load(trained_model_path_abs, map_location=device))
        
        # Set the model to evaluation mode (crucial for inference)
        model.eval() 
        
        _visual_deepfake_model_instance = model
        _detection_device = device
        
        print(f"Deepfake model '{model_name}' loaded successfully from {model_path}.", file=sys.stderr)
        
        # MediaPipe setup for fast and robust face detection.
        # Assign directly to the global variables
        _mp_face_detection = mp.solutions.face_detection # <-- CHANGED: Assign to global variable
        _mp_drawing = mp.solutions.drawing_utils         # <-- CHANGED: Assign to global variable
        
        print("Trained Deepfake Detector and MediaPipe setup complete.", file=sys.stderr)
    except FileNotFoundError as e:
        print(f"ERROR: {e}. Please ensure your trained model file is in the correct location.", file=sys.stderr)
        _visual_deepfake_model_instance = None
        _detection_device = None
        _mp_face_detection = None # <-- ADDED: Set to None on error
        _mp_drawing = None        # <-- ADDED: Set to None on error
        return None, None
    except Exception as e:
        print(f"ERROR: Failed to load trained PyTorch model or MediaPipe: {e}", file=sys.stderr)
        _visual_deepfake_model_instance = None
        _detection_device = None
        _mp_face_detection = None # <-- ADDED: Set to None on error
        _mp_drawing = None        # <-- ADDED: Set to None on error
        return None, None

# Load LLM for fingerprinting (existing code, assumes distilbert-base-uncased)
def _load_llm_fingerprinting_model():
    global _llm_tokenizer, _llm_model
    print("Initializing LLM fingerprinting components (conceptual)...", file=sys.stderr)
    try:
        _llm_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        _llm_model = AutoModel.from_pretrained("distilbert-base-uncased")
        print("LLM fingerprinting components loaded.", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Error loading LLM fingerprinting components: {e}", file=sys.stderr)
        _llm_tokenizer = None
        _llm_model = None

# --- GLOBAL MODEL LOADING ON IMPORT ---
# These functions will be called once when ls_zlf.py is imported by main.py
# This ensures models are loaded into memory when the server starts.
_load_deepfake_model(model_path="efficientnet_b0_deepfake.pth") # Load your deepfake model here
_load_llm_fingerprinting_model() # Load LLM components


# --- Image Preprocessing Transforms for Deepfake Model ---
# These should match your val_test_transform from your training notebook
# Assuming input image size is 224x224 and normalization is standard ImageNet means/stds
image_transform = transforms.Compose([
    transforms.Resize((224, 224)), # Resize faces to the model's expected input
    transforms.ToTensor(), # Convert PIL Image to PyTorch Tensor
    transforms.Normalize( # Normalize with ImageNet stats
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --- Helper Functions (Visual Deepfake Analysis) ---

# 1. Prediction Function for a Single Image (Face)
def predict_image_deepfake(model: nn.Module, device: torch.device, pil_image: Image.Image):
    """
    Predicts deepfake probability for a single PIL Image (e.g., a cropped face).
    Accepts model and device as arguments.
    Returns probability of being FAKE (0.0 to 1.0) and predicted label ("FAKE" or "REAL").
    """
    if model is None:
        print("Deepfake model not provided to predict_image_deepfake.", file=sys.stderr)
        return 0.5, "ERROR: Model not loaded"

    # Apply transformations and move to device
    input_tensor = image_transform(pil_image).unsqueeze(0).to(device)

    model.eval() # Ensure model is in evaluation mode for prediction
    with torch.no_grad(): # Disable gradient calculation for inference
        output = model(input_tensor)
        # Use sigmoid as your model likely outputs logits (raw scores) for binary classification
        probability = torch.sigmoid(output).item() # Get probability for the FAKE class (assuming 1 output neuron)

    label = "FAKE" if probability > 0.5 else "REAL" # Standard 0.5 threshold
    return probability, label

# 2. Face Detection and Cropping (using MediaPipe)
def detect_and_crop_faces(image_np):
    """
    Detects faces in a NumPy image array (BGR format, typically from OpenCV) and returns cropped PIL Images.
    Also returns the original image with bounding boxes drawn (BGR format) for visualization.
    """
    # Check if MediaPipe components loaded globally
    if _mp_face_detection is None or _mp_drawing is None: 
        print("MediaPipe Face Detection not loaded. Skipping face detection.", file=sys.stderr)
        return [], image_np # Return empty list of faces and original image

    faces = []
    annotated_image_np = image_np.copy() # Create a copy for drawing on

    # MediaPipe requires RGB images
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB) 

    # Use MediaPipe FaceDetection pipeline (using global _mp_face_detection)
    with _mp_face_detection.FaceDetection( 
        model_selection=1, # 0 for short-range, 1 for full-range (better for varied distances)
        min_detection_confidence=0.5 # Confidence threshold for detection
    ) as face_detection:
        results = face_detection.process(image_rgb)

        if results.detections:
            for detection in results.detections:
                # Draw face detection annotations on the annotated copy of the image (using global _mp_drawing).
                _mp_drawing.draw_detection(annotated_image_np, detection) 

                # Get bounding box coordinates relative to the image
                bbox_c = detection.location_data.relative_bounding_box
                ih, iw, _ = image_np.shape # Image height, width
                
                # Convert relative coordinates to absolute pixel coordinates
                x = int(bbox_c.xmin * iw)
                y = int(bbox_c.ymin * ih)
                w = int(bbox_c.width * iw)
                h = int(bbox_c.height * ih)

                # Expand bounding box slightly for better cropping (optional, but common)
                expand_px = 20 # Expand by 20 pixels on each side
                x_exp = max(0, x - expand_px)
                y_exp = max(0, y - expand_px)
                w_exp = min(iw - x_exp, w + 2 * expand_px)
                h_exp = min(ih - y_exp, h + 2 * expand_px)


                if w_exp > 0 and h_exp > 0: # Ensure valid crop dimensions
                    cropped_face_np = image_np[y_exp:y_exp+h_exp, x_exp:x_exp+w_exp]
                    # Convert cropped face (BGR from OpenCV) to PIL Image (RGB) for PyTorch model
                    cropped_face_pil = Image.fromarray(cv2.cvtColor(cropped_face_np, cv2.COLOR_BGR2RGB))
                    faces.append(cropped_face_pil)
    return faces, annotated_image_np

# 3. Helper for converting NumPy array to Base64 (for frontend display)
def numpy_to_base64_image(image_np):
    """Converts a NumPy array image (BGR) to a base64 encoded PNG string."""
    if image_np is None:
        return None
    # Ensure the image is in a displayable format (e.g., uint8, range 0-255)
    if image_np.dtype != np.uint8:
        image_np = (image_np * 255).astype(np.uint8) 
    
    # Convert from BGR (OpenCV default) to RGB for PIL, then save as PNG
    pil_image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='PNG') # Save as PNG for transparency/quality
    encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    return encoded_img

# 4. Grad-CAM Implementation (for visual explanations)
def generate_grad_cam(model: nn.Module, device: torch.device, pil_image: Image.Image, target_layer_name='_conv_head'): # Adjusted default target layer name
    """
    Generates a Grad-CAM heatmap for a given PIL image and model,
    and overlays it on the original image.
    Returns the base64 encoded image with heatmap.
    """
    if model is None or not isinstance(model, nn.Module): # Ensure model is a PyTorch module
        print("Grad-CAM: Valid PyTorch model not loaded.", file=sys.stderr)
        return None

    model.eval() # Ensure model is in evaluation mode

    # Find the target layer for Grad-CAM
    target_layer = None
    try:
        # Common approach to get the last convolutional layer before pooling/classifier in EfficientNet
        # For efficientnet_pytorch, _conv_head or the last block's _project_conv/_conv_ow are common targets
        for name, module in model.named_modules(): # Iterate through all named modules in the model
            if name == target_layer_name: 
                target_layer = module
                break
        
        # Fallback if specific target layer name isn't found
        if target_layer is None:
            if hasattr(model.backbone, '_conv_head'):
                target_layer = model.backbone._conv_head # Common last conv in EfficientNet backbone
            elif hasattr(model.backbone, '_blocks') and len(model.backbone._blocks) > 0:
                last_block = model.backbone._blocks[-1]
                if hasattr(last_block, '_project_conv'):
                     target_layer = last_block._project_conv
                elif hasattr(last_block, '_conv_ow'):
                    target_layer = last_block._conv_ow
            
        if target_layer is None:
            print(f"Warning: Could not find a suitable target layer for Grad-CAM. Tried '{target_layer_name}' and common fallbacks. Returning None.", file=sys.stderr)
            return None

    except Exception as e:
        print(f"Error finding target layer for Grad-CAM: {e}", file=sys.stderr)
        return None

    # Preprocess the image for the model
    # Use the same image_transform used for prediction
    input_tensor = image_transform(pil_image).unsqueeze(0).to(device)

    # Define targets for Grad-CAM (e.g., the predicted class, 0 for FAKE in binary classification)
    # Assuming output is sigmoid probability, so target 0 means target the 'FAKE' class output
    # ClassifierOutputTarget expects raw logits, not probabilities
    # If your model outputs logits before sigmoid, this is fine. If it outputs probability, may need adjustment.
    targets = [ClassifierOutputTarget(category=0)] # Target the output for the 'FAKE' class (index 0)

    # Initialize GradCAM
    cam = GradCAM(model=model, target_layers=[target_layer]) # REMOVED use_cuda argument
    # Generate heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :] # Get the heatmap for the first image in batch

    # Overlay heatmap on original image (resize PIL image for consistency)
    rgb_img_np = np.array(pil_image.resize((224, 224))) / 255.0 # Scale to 0-1 for overlay
    cam_image_np = show_cam_on_image(rgb_img_np, grayscale_cam, use_rgb=True)

    # Convert the result (NumPy array, RGB float 0-1) to base64 encoded PNG string
    # numpy_to_base64_image expects BGR uint8, so convert back
    cam_image_bgr = cv2.cvtColor((cam_image_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return numpy_to_base64_image(cam_image_bgr)


# --- Helper Functions (LLM Fingerprinting - existing, kept for now) ---
# Assuming these are correct from previous steps
# This might be replaced by a more professional LLM detector later (Goal 2)

# Simulated LLM signatures based on typical characteristics
LLM_SIGNATURES = {
    "Likely AI (General)": {
        "phrases": [
            "as an AI language model",
            "I cannot fulfill this request",
            "I do not have personal opinions",
            "I am a large language model",
            "it is important to note that",
            "in conclusion, it is evident that"
        ],
        "description": "Text contains common phrasing characteristic of large language models.",
        "confidence_score": 0.8
    },
    "Likely Human-Written": {
        "phrases": [], # No specific phrases to define human
        "description": "Text does not strongly match known AI patterns (needs more sophisticated analysis to confirm human origin).",
        "confidence_score": 0.2
    }
}

def identify_llm_origin_simplified(text_content: str):
    if not text_content or not text_content.strip():
        return {"llm_origin": "N/A", "confidence": 0, "reason": "No text provided for LLM analysis."}
    
    text_lower = text_content.lower()
    
    for llm_type, signature in LLM_SIGNATURES.items():
        if llm_type == "Likely Human-Written":
            continue 

        for phrase in signature["phrases"]:
            if phrase in text_lower:
                return {
                    "llm_origin": llm_type,
                    "confidence": signature["confidence_score"],
                    "reason": f"{signature['description']} (Found phrase: '{phrase}')"
                }
    
    return {
        "llm_origin": "Human/Uncertain",
        "confidence": LLM_SIGNATURES["Likely Human-Written"]["confidence_score"],
        "reason": LLM_SIGNATURES["Likely Human-Written"]["description"]
    }


# --- Main LS-ZLF Analysis Function ---
async def analyze_ls_zlf(media_paths: dict):
    results = {
        "deepfake_analysis": {
            "deepfake_detected": False, 
            "reason": "No visual media for deepfake analysis.",
            "probability": 0.0,
            "explanation_image_b64": None,
            "faces_detected_count": 0
        },
        "llm_origin_analysis": {}
    }

    # Process Video/Image for Visual Deepfake Detection
    visual_media_path = media_paths.get("video") # Prioritize video
    if not visual_media_path and media_paths.get("image"): # Fallback to image if provided
        visual_media_path = media_paths.get("image") # Assuming "image" can be uploaded (not in current UI)
    
    # Check if visual deepfake model is loaded AND there's visual media
    if visual_media_path and os.path.exists(visual_media_path) and _visual_deepfake_model_instance and _detection_device:
        print(f"DEBUG: Performing visual deepfake analysis on: {visual_media_path}", file=sys.stderr)
        try:
            # Read the first frame of the video or the image file
            if visual_media_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                cap = cv2.VideoCapture(visual_media_path)
                ret, frame_np = cap.read() # Read one frame
                cap.release()
                if not ret:
                    results["deepfake_analysis"]["reason"] = "Could not read video frame for visual analysis."
                    print("DEBUG: Could not read video frame for visual analysis.", file=sys.stderr)
                    return results # Exit early if frame cannot be read
            else: # Assume it's an image file
                frame_np = cv2.imread(visual_media_path)
            
            if frame_np is None: # Check if imread or cap.read failed
                results["deepfake_analysis"]["reason"] = "Could not load image/video frame for visual analysis (check file format)."
                print("DEBUG: Could not load image/video frame for visual analysis.", file=sys.stderr)
                return results # Exit early if frame is None

            # Perform face detection and get annotated image
            faces_pil, annotated_frame_np = detect_and_crop_faces(frame_np)
            results["deepfake_analysis"]["faces_detected_count"] = len(faces_pil)
            print(f"DEBUG: Found {len(faces_pil)} faces.", file=sys.stderr)

            if len(faces_pil) > 0:
                # Take the first detected face for prediction and explanation
                face_pil = faces_pil[0] 
                
                # Predict deepfake probability
                deepfake_prob, deepfake_label = predict_image_deepfake(_visual_deepfake_model_instance, _detection_device, face_pil)
                results["deepfake_analysis"]["probability"] = deepfake_prob
                results["deepfake_analysis"]["deepfake_detected"] = (deepfake_label == "FAKE") # Set based on model output
                results["deepfake_analysis"]["reason"] = f"Face detected. Predicted as {deepfake_label} with probability {deepfake_prob:.2f}."

                # Generate Grad-CAM heatmap
                print("DEBUG: Attempting Grad-CAM generation...", file=sys.stderr)
                # target_layer_name '_conv_head' is common in efficientnet_pytorch models
                explanation_b64 = generate_grad_cam(_visual_deepfake_model_instance, _detection_device, face_pil, target_layer_name='_conv_head') 
                
                if explanation_b64:
                    results["deepfake_analysis"]["explanation_image_b64"] = explanation_b64
                    print("DEBUG: Grad-CAM explanation generated and encoded.", file=sys.stderr)
                else:
                    print("DEBUG: Grad-CAM explanation failed or target layer not found. Providing face detection image.", file=sys.stderr)
                    # Fallback to just sending the original annotated frame if Grad-CAM fails
                    results["deepfake_analysis"]["explanation_image_b64"] = numpy_to_base64_image(annotated_frame_np)
                    if results["deepfake_analysis"]["explanation_image_b64"]:
                        results["deepfake_analysis"]["reason"] += " (Grad-CAM failed, showing face detection instead)"
                    else:
                        results["deepfake_analysis"]["reason"] += " (Failed to generate any visual explanation)"

            else: # No faces detected in the visual media
                results["deepfake_analysis"]["reason"] = "No faces detected in video/image for visual deepfake analysis."
                # Still provide a base64 image of the frame with potential bounding boxes, for visual context
                results["deepfake_analysis"]["explanation_image_b64"] = numpy_to_base64_image(annotated_frame_np)

        except Exception as e:
            # Catch any unhandled errors during visual analysis
            print(f"ERROR: Visual deepfake analysis failed with unexpected error: {e}", file=sys.stderr)
            results["deepfake_analysis"]["reason"] = f"Visual deepfake analysis failed: {str(e)}"
            # Fallback to general reason if visual analysis completely fails
            if "deepfake_detected" not in results["deepfake_analysis"]: 
                results["deepfake_analysis"]["deepfake_detected"] = False
    else:
        # Fallback if visual deepfake model is not loaded or no visual media provided
        if not _visual_deepfake_model_instance:
            results["deepfake_analysis"]["reason"] = "Visual deepfake model not loaded (check startup logs for errors)."
        elif not visual_media_path:
            results["deepfake_analysis"]["reason"] = "No visual media provided for deepfake analysis."
        # If no visual_media_path, explanation_image_b64 should be None, set explicitly
        results["deepfake_analysis"]["explanation_image_b64"] = None


    # LLM origin analysis (existing logic)
    # This will be upgraded later in Goal 2
    if media_paths.get("text") and os.path.exists(media_paths["text"]):
        with open(media_paths["text"], 'r', encoding='utf-8') as f:
            text_content = f.read()
            results["llm_origin_analysis"] = identify_llm_origin_simplified(text_content)
    else:
        results["llm_origin_analysis"] = {"llm_origin": "N/A", "confidence": 0, "reason": "No text provided for LLM analysis."}

    return results