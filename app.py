import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import os
import mlflow
from data_handler import get_transforms
import glob
import logging
import sys
import warnings
import mediapipe as mp
import numpy as np
import cv2
from feature_visualization import get_feature_maps, display_feature_maps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='timm')
warnings.filterwarnings('ignore', category=UserWarning, module='huggingface_hub')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.serialization')

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Set page config
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .stImage > img {
            max-height: 400px;
            width: auto;
        }
        div.stMarkdown {
            max-width: 100%;
        }
        div[data-testid="column"] {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 10px;
            margin: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# Constants
MODEL_IMAGE_SIZES = {
    "xception": 299,
    "efficientnet": 300,
    "swin": 224,
    "cross_attention": 224,
    "two_stream": 224,
    "cnn_transformer": 224
}

def extract_face(image, padding=0.1):
    """Extract face from image using MediaPipe with padding"""
    # Convert PIL Image to cv2 format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    height, width = img_cv.shape[:2]
    
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        # Convert the BGR image to RGB
        results = face_detection.process(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        
        if not results.detections:
            return None, None
        
        # Get the first face detection
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        
        # Calculate padding
        pad_w = int(bbox.width * width * padding)
        pad_h = int(bbox.height * height * padding)
        
        # Convert relative coordinates to absolute with padding
        x = int(bbox.xmin * width) - pad_w
        y = int(bbox.ymin * height) - pad_h
        w = int(bbox.width * width) + (2 * pad_w)
        h = int(bbox.height * height) + (2 * pad_h)
        
        # Ensure coordinates are within image bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, width - x)
        h = min(h, height - y)
        
        # Extract face region with padding
        face_region = img_cv[y:y+h, x:x+w]
        
        # Convert back to RGB for PIL
        face_region_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_region_rgb)
        
        # Create visualization
        img_cv_viz = img_cv.copy()
        cv2.rectangle(img_cv_viz, (x, y), (x+w, y+h), (0, 255, 0), 2)
        img_viz = cv2.cvtColor(img_cv_viz, cv2.COLOR_BGR2RGB)
        
        return face_pil, Image.fromarray(img_viz)

def resize_image_for_display(image, max_size=400):
    """Resize image for display while maintaining aspect ratio"""
    width, height = image.size
    if width > height:
        if width > max_size:
            ratio = max_size / width
            new_size = (max_size, int(height * ratio))
    else:
        if height > max_size:
            ratio = max_size / height
            new_size = (int(width * ratio), max_size)
    
    if width > max_size or height > max_size:
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image

def process_image(image, model_type):
    """Process uploaded image using the same transforms as training"""
    try:
        # Get correct image size for model
        img_size = MODEL_IMAGE_SIZES.get(model_type, 224)
        transform = get_transforms(img_size)
        transformed_image = transform(image).unsqueeze(0)
        logger.info(f"Successfully processed image for {model_type} (size: {img_size})")
        return transformed_image
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return None

def load_model(model_path, model_type):
    """Load a model from converted state dict"""
    try:
        # Initialize model based on type
        model = None
        try:
            if model_type == "xception":
                from train_xception import DeepfakeXception
                model = DeepfakeXception()
            elif model_type == "efficientnet":
                from train_efficientnet import DeepfakeEfficientNet
                model = DeepfakeEfficientNet()
            elif model_type == "swin":
                from train_swin import DeepfakeSwin
                model = DeepfakeSwin()
            elif model_type == "cross_attention":
                from train_cross_attention import DeepfakeCrossAttention
                model = DeepfakeCrossAttention()
            elif model_type == "two_stream":
                from train_two_stream import DeepfakeTwoStream
                model = DeepfakeTwoStream()
            elif model_type == "cnn_transformer":
                from train_cnn_transformer import DeepfakeCNNTransformer
                model = DeepfakeCNNTransformer()
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            return None
        
        if model is None:
            logger.warning(f"Could not initialize model of type {model_type}")
            return None
        
        # Load state dict with weights_only=True to avoid pickle security warning
        try:
            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Successfully loaded state dict for {model_type}")
        except Exception as e:
            logger.error(f"Error loading state dict: {str(e)}")
            return None
            
        model.eval()
        logger.info(f"Successfully initialized {model_type} model")
        
        return model
    except Exception as e:
        logger.error(f"Error in load_model: {str(e)}")
        return None

def format_confidence(confidence):
    """Format confidence score with color based on value"""
    if confidence >= 0.9:
        color = "red" if confidence >= 0.95 else "orange"
    else:
        color = "green" if confidence >= 0.7 else "blue"
    return f'<span style="color: {color}; font-weight: bold;">{confidence:.1%}</span>'

def main():
    st.title("🔍 Deepfake Detection System")
    st.write("Upload an image to check if it's real or fake using multiple deep learning models.")
    
    # Sidebar information
    with st.sidebar:
        st.write("### System Information")
        st.write(f"Device: CPU (GPU not available)")
        st.write("### Model Information")
        for model_type, size in MODEL_IMAGE_SIZES.items():
            st.write(f"- {model_type.upper()}: {size}x{size}")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Load and display the image
            image = Image.open(uploaded_file).convert('RGB')
            
            # Extract face
            face_image, viz_image = extract_face(image)
            
            if face_image is None:
                st.error("No face detected in the image. Please upload an image containing a clear face.")
                return
            
            # Create two columns for image and info
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Display original image with face detection
                st.write("### Original Image with Face Detection")
                st.image(viz_image, use_container_width=True)
                
                # Display extracted face
                st.write("### Extracted Face")
                display_face = resize_image_for_display(face_image)
                st.image(display_face, use_container_width=True)
                st.write(f"Face size: {face_image.size[0]}x{face_image.size[1]}")
            
            with col2:
                # Load all converted models
                converted_dir = "converted_models"
                if not os.path.exists(converted_dir):
                    st.error("Please run convert_models.py first to convert the models!")
                    return
                
                # Find all converted model files
                model_files = glob.glob(os.path.join(converted_dir, "*_converted.pth"))
                if not model_files:
                    st.error("No converted models found! Please run convert_models.py first.")
                    return
                
                st.write(f"### Model Predictions")
                
                # Create tabs for predictions and visualizations
                pred_tab, viz_tab = st.tabs(["Predictions", "Feature Visualizations"])
                
                with pred_tab:
                    # Create columns for results
                    cols = st.columns(2)
                    col_idx = 0
                    models_loaded = False
                    
                    progress_bar = st.progress(0)
                    predictions_data = []  # Store predictions for later use
                    
                    for i, model_path in enumerate(model_files):
                        # Get model type from filename
                        model_type = os.path.basename(model_path).replace("_converted.pth", "")
                        
                        with st.spinner(f'Processing with {model_type.upper()}...'):
                            model = load_model(model_path, model_type)
                            if model is None:
                                continue
                            
                            models_loaded = True
                            
                            # Process image for this specific model
                            processed_image = process_image(face_image, model_type)
                            if processed_image is None:
                                st.error(f"Failed to process image for {model_type}")
                                continue
                            
                            # Make prediction
                            try:
                                with torch.no_grad():
                                    output = model(processed_image)
                                    probability = torch.sigmoid(output).item()
                                    prediction = "FAKE" if probability > 0.5 else "REAL"
                                    confidence = probability if prediction == "FAKE" else 1 - probability
                                
                                # Store prediction data
                                predictions_data.append({
                                    'model_type': model_type,
                                    'prediction': prediction,
                                    'confidence': confidence,
                                    'model': model,
                                    'processed_image': processed_image
                                })
                                
                                # Display results in card format
                                with cols[col_idx % 2]:
                                    st.markdown(f"""
                                    <div style='padding: 15px; border-radius: 10px; border: 1px solid #ddd; margin: 5px 0;'>
                                        <h4>{model_type.upper()}</h4>
                                        <p>Prediction: <strong>{prediction}</strong><br>
                                        Confidence: {format_confidence(confidence)}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                col_idx += 1
                                
                            except Exception as e:
                                st.error(f"Error making prediction with {model_type}: {str(e)}")
                        
                        progress_bar.progress((i + 1) / len(model_files))
                
                with viz_tab:
                    if predictions_data:
                        # Add model selector
                        selected_model = st.selectbox(
                            "Select Model for Visualization",
                            options=[p['model_type'].upper() for p in predictions_data],
                            format_func=lambda x: f"{x} Model"
                        )
                        
                        # Get selected model data
                        model_data = next(p for p in predictions_data if p['model_type'].upper() == selected_model)
                        
                        # Get and display feature maps
                        visualizations = get_feature_maps(
                            model_data['model'],
                            model_data['model_type'],
                            model_data['processed_image']
                        )
                        display_feature_maps(face_image, visualizations)
                    else:
                        st.warning("No models available for visualization.")
                
                if not models_loaded:
                    st.error("No models could be loaded! Please check the model files.")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Error in main app: {str(e)}")

if __name__ == "__main__":
    main()