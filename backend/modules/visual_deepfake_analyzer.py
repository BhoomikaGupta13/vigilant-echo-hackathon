# backend/modules/visual_deepfake_analyzer.py

import cv2
import numpy as np
import os
import io
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import sys
import base64
import mediapipe as mp # For face detection
import dlib # For Dlib face detector and shape predictor
from imutils import face_utils # For Dlib landmark utilities
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

# --- Configuration ---
# Path to your trained model weights
MODEL_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'models', 'efficientnet_b0_deepfake.pth')
# Path to Dlib's shape predictor model
PREDICTOR_PATH = os.path.join(os.path.dirname(__file__), 'models', 'shape_predictor_68_face_landmarks.dat')

# --- 1. Deepfake Detector Model Definition ---
class DeepfakeDetectorModel(torch.nn.Module):
    def __init__(self):
        super(DeepfakeDetectorModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device for DeepfakeDetectorModel: {self.device}")

        # Load pre-trained EfficientNetB0
        # EfficientNet_B0_Weights.IMAGENET1K_V1 provides the standard ImageNet weights
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Modify the classifier head for binary deepfake classification (1 output feature)
        # EfficientNet has a 'classifier' module which is a Sequential(Linear, Dropout, Linear)
        # We replace the last Linear layer (index 1 in the classifier Sequential)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = torch.nn.Linear(num_ftrs, 1) # Output a single logit for binary classification

        self.model.to(self.device)

        # Load your trained state_dict
        if os.path.exists(MODEL_WEIGHTS_PATH):
            try:
                state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=self.device)
                
                # Adjust state_dict keys if saved from DataParallel
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        new_state_dict[k[7:]] = v # remove 'module.' prefix
                    else:
                        new_state_dict[k] = v

                self.model.load_state_dict(new_state_dict)
                self.model.eval() # Set model to evaluation mode
                logger.info(f"Deepfake Detection Model loaded successfully from {MODEL_WEIGHTS_PATH}.")
            except Exception as e:
                logger.error(f"Error loading Deepfake Detection Model weights from {MODEL_WEIGHTS_PATH}: {e}", exc_info=True)
                logger.warning("Using untrained EfficientNetB0 for deepfake detection due to loading error.")
                # Keep model in train mode if weights failed to load, or handle as appropriate
                self.model.train() # Default to train mode if loading fails, or just proceed without weights
        else:
            logger.warning(f"Deepfake Detection Model weights not found at {MODEL_WEIGHTS_PATH}. Using untrained EfficientNetB0.")
            self.model.train() # Default to train mode if weights not found

    def forward(self, x):
        return torch.sigmoid(self.model(x)) # Apply sigmoid for probability output


# --- 2. Face Extraction and Preprocessing ---
class FaceExtractor:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize Dlib face detector and landmark predictor
        try:
            if not os.path.exists(PREDICTOR_PATH):
                logger.error(f"Error: Dlib shape predictor not found at {PREDICTOR_PATH}.")
                logger.error("Please download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 and place it in the 'models/' directory.")
                self.detector = None
                self.predictor = None
            else:
                self.detector = dlib.get_frontal_face_detector()
                self.predictor = dlib.shape_predictor(PREDICTOR_PATH)
        except Exception as e:
            logger.error(f"Failed to initialize Dlib: {e}", exc_info=True)
            self.detector = None
            self.predictor = None

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)), # EfficientNetB0 input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet normalization
        ])

    def detect_and_extract_face(self, image_rgb):
        """
        Detects faces in an image using MediaPipe or Dlib and extracts the largest face.
        Returns the cropped face image (OpenCV format) and its bounding box.
        """
        h, w, _ = image_rgb.shape
        face_img = None
        bbox = None

        # Try MediaPipe first (generally faster for face detection)
        results = self.face_detection.process(image_rgb)
        if results.detections:
            for detection in results.detections:
                mp_bbox_data = detection.location_data.relative_bounding_box
                xmin = int(mp_bbox_data.xmin * w)
                ymin = int(mp_bbox_data.ymin * h)
                width = int(mp_bbox_data.width * w)
                height = int(mp_bbox_data.height * h)

                # Expand bounding box slightly for better context
                padding = 0.1 # 10% padding
                px = int(width * padding)
                py = int(height * padding)
                xmin = max(0, xmin - px)
                ymin = max(0, ymin - py)
                xmax = min(w, xmin + width + 2 * px)
                ymax = min(h, ymin + height + 2 * py)

                face_img = image_rgb[ymin:ymax, xmin:xmax]
                bbox = (xmin, ymin, xmax, ymax) # (xmin, ymin, xmax, ymax)
                break # Take the first detected face (usually the largest/most prominent)
        
        # Fallback to Dlib if MediaPipe finds no faces (or if it's not initialized)
        if face_img is None and self.detector is not None:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            rects = self.detector(gray, 0)
            if len(rects) > 0:
                rect = rects[0] # Take the first detected face by Dlib
                xmin, ymin, xmax, ymax = rect.left(), rect.top(), rect.right(), rect.bottom()
                
                # Expand bounding box slightly
                padding = 0.1
                px = int((xmax - xmin) * padding)
                py = int((ymax - ymin) * padding)
                xmin = max(0, xmin - px)
                ymin = max(0, ymin - py)
                xmax = min(w, xmax + px)
                ymax = min(h, ymax + py)

                face_img = image_rgb[ymin:ymax, xmin:xmax]
                bbox = (xmin, ymin, xmax, ymax)

        return face_img, bbox

    def preprocess_face(self, face_img):
        """Applies necessary transformations to the face image for model input."""
        if face_img is None:
            return None
        return self.transform(face_img).unsqueeze(0) # Add batch dimension


# --- 3. Explanation Generator (Grad-CAM) ---
class ExplanationGenerator:
    def __init__(self, deepfake_detector_model_instance):
        self.deepfake_detector_model = deepfake_detector_model_instance.model
        self.device = deepfake_detector_model_instance.device

        # For EfficientNet-B0, _blocks[-1] (the last block) is a good common choice.
        # This targets the last convolutional layer before the classification head.
        self.target_layers = [self.deepfake_detector_model.features[-1]] # Correct target layer for torchvision EfficientNet
        logger.info(f"Using EfficientNet target layer for Grad-CAM: {self.target_layers[0].__class__.__name__}")

    def generate_explanation(self, original_image_pil: Image.Image, input_tensor: torch.Tensor, bbox: tuple) -> str:
        """
        Generates a Grad-CAM explanation and overlays it on the original image (or cropped face).
        Returns the base64 encoded image with heatmap.
        Args:
            original_image_pil (PIL.Image.Image): The original PIL image of the frame.
            input_tensor (torch.Tensor): The preprocessed face tensor for model input.
            bbox (tuple): Bounding box (xmin, ymin, xmax, ymax) of the detected face in the original image.
        Returns:
            str: Base64 encoded PNG image with heatmap.
        """
        if input_tensor is None or bbox is None:
            return None

        # Create GradCAM instance
        cam = GradCAM(model=self.deepfake_detector_model, target_layers=self.target_layers, use_cuda=torch.cuda.is_available())

        # Define target for CAM (probability of deepfake, typically 0 for 'real' or 1 for 'fake')
        # Since our model outputs a probability (sigmoid), we want to get CAM for that specific output.
        # If your target is 'deepfake', then target_category should be 0 (index for fake class assuming binary output and 1 being real).
        # For a single output logit/probability, `ClassifierOutputTarget` is good.
        # Here, we want explanation for the 'deepfake' class (probability closer to 1).
        # targets = [ClassifierOutputTarget(0)] # For a 2-class model (0:real, 1:fake), if 1 is deepfake, target 1.
        # For a single sigmoid output, we generally want the activation that leads to higher probability.
        # Let's assume we want to visualize what makes it 'deepfake', so target the output.
        # A simpler way without explicit target index for single output:
        targets = None # No specific target means it targets the highest logit (which is our single output)

        # Generate the heatmap. Ensure input_tensor is on the correct device.
        grayscale_cam = cam(input_tensor=input_tensor.to(self.device), targets=targets)
        
        # In this example, GradCAM returns a batch of 1.
        grayscale_cam = grayscale_cam[0, :]

        # Normalize the original input image for overlay
        # `show_cam_on_image` expects float32 image with pixel values in range [0, 1]
        original_img_np = np.array(original_image_pil) / 255.0

        # Overlay heatmap on the original image within the detected face region
        # First, create a blank image to draw heatmap on, matching original image size
        heatmap_overlay_img = np.zeros_like(original_img_np)
        
        # Get the cropped face image in 0-1 float range
        xmin, ymin, xmax, ymax = bbox
        face_region_orig = original_img_np[ymin:ymax, xmin:xmax]

        # Resize the grayscale_cam to match the face region dimensions
        # Use skimage.transform.resize for higher quality resizing
        from skimage.transform import resize
        resized_cam = resize(grayscale_cam, (face_region_orig.shape[0], face_region_orig.shape[1]), anti_aliasing=True)

        # Apply the heatmap only to the face region
        cam_image = show_cam_on_image(face_region_orig, resized_cam, use_rgb=True)
        
        # Place the CAM image back into the full frame
        heatmap_overlay_img[ymin:ymax, xmin:xmax] = cam_image
        
        # Convert the NumPy array to a PIL Image
        # Multiply by 255 and convert to uint8 for PIL and then PNG encoding
        img_pil = Image.fromarray((heatmap_overlay_img * 255).astype(np.uint8))
        
        # Save to a byte stream and base64 encode
        buffered = io.BytesIO()
        img_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str


# --- Main Analysis Function ---
async def analyze_visual_deepfake(video_path: str):
    logger.info(f"Starting visual deepfake analysis for video: {video_path}")
    
    detector_model = DeepfakeDetectorModel()
    face_extractor = FaceExtractor()
    explanation_gen = ExplanationGenerator(detector_model)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error: Could not open video file {video_path}")
        return {
            "overall_deepfake_probability": 0.0,
            "detailed_explanation": "Could not process video file.",
            "frame_wise_explanations": [],
            "heatmaps": []
        }

    frame_count = 0
    deepfake_probabilities = []
    frame_explanations = []
    heatmaps_base64 = [] # Store base64 encoded heatmaps

    # Process every Nth frame to speed up (e.g., every 10th frame)
    frame_skip = 5 # Analyze every 5th frame
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Convert BGR to RGB for MediaPipe and PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_frame_pil = Image.fromarray(frame_rgb)

        # 1. Face Detection and Extraction
        face_img_cv2, bbox = face_extractor.detect_and_extract_face(frame_rgb)

        if face_img_cv2 is not None:
            # 2. Preprocess Face for Model Input
            input_tensor = face_extractor.preprocess_face(face_img_cv2)

            if input_tensor is not None:
                with torch.no_grad(): # No need to compute gradients during inference
                    input_tensor = input_tensor.to(detector_model.device)
                    prediction = detector_model(input_tensor)
                    prob = prediction.item() # Get the single probability value

                deepfake_probabilities.append(prob)

                # Generate Heatmap
                heatmap_b64 = explanation_gen.generate_explanation(original_frame_pil, input_tensor, bbox)
                heatmaps_base64.append(heatmap_b64)

                # Generate Frame-wise Explanation
                explanation_text = "Potentially deepfake" if prob > 0.5 else "Likely real"
                frame_explanations.append({
                    "frame_number": frame_count,
                    "probability": prob,
                    "explanation": explanation_text
                })
            else:
                logger.warning(f"Could not preprocess face in frame {frame_count}.")
                heatmaps_base64.append(None) # Append None if no heatmap could be generated
        else:
            logger.warning(f"No face detected in frame {frame_count}.")
            deepfake_probabilities.append(0.0) # Assume real if no face, or handle as unknown
            frame_explanations.append({
                "frame_number": frame_count,
                "probability": 0.0, # Or some 'N/A' value
                "explanation": "No face detected"
            })
            heatmaps_base64.append(None) # Append None if no heatmap could be generated


    cap.release()

    overall_prob = np.mean(deepfake_probabilities) if deepfake_probabilities else 0.0
    detailed_explanation = "No specific deepfake indicators found."
    if overall_prob > 0.7:
        detailed_explanation = "High likelihood of deepfake manipulation detected across frames."
    elif overall_prob > 0.5:
        detailed_explanation = "Moderate likelihood of deepfake manipulation detected in some frames."
    elif overall_prob > 0.3:
        detailed_explanation = "Low likelihood of deepfake manipulation, some minor indicators."
    
    logger.info(f"Visual analysis complete. Overall probability: {overall_prob:.4f}")

    return {
        "overall_deepfake_probability": overall_prob,
        "detailed_explanation": detailed_explanation,
        "frame_wise_explanations": frame_explanations,
        "heatmaps": heatmaps_base64 # Return base64 encoded heatmap images
    }