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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='timm')
warnings.filterwarnings('ignore', category=UserWarning, module='huggingface_hub')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.serialization')

# Set page config
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🔍",
    layout="wide"
)

# Constants
IMAGE_SIZE = 299  # Xception needs 299, others will be resized
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_image_size(model_type):
    """Get the correct image size for each model"""
    if model_type == "xception":
        return 299
    elif model_type == "efficientnet":
        return 300
    elif model_type == "swin":
        return 224
    else:
        return 224

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
        
        # Load state dict
        try:
            state_dict = torch.load(model_path, map_location='cpu')
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

def process_image(image, model_type):
    """Process uploaded image using the same transforms as training"""
    try:
        # Get correct image size for model
        img_size = get_image_size(model_type)
        transform = get_transforms(img_size)
        transformed_image = transform(image).unsqueeze(0)  # No need to send to device here
        logger.info(f"Successfully processed image for {model_type} (size: {img_size})")
        return transformed_image
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return None

def main():
    st.title("🔍 Deepfake Detection System")
    st.write("Upload an image to check if it's real or fake using multiple deep learning models.")
    
    # Display device information
    st.sidebar.write(f"Using device: CPU (GPU not available)")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
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
            
            st.write(f"Found {len(model_files)} converted models")
            
            # Create columns for results
            cols = st.columns(3)
            col_idx = 0
            models_loaded = False
            
            progress_bar = st.progress(0)
            for i, model_path in enumerate(model_files):
                # Get model type from filename
                model_type = os.path.basename(model_path).replace("_converted.pth", "")
                
                with st.spinner(f'Loading {model_type} model...'):
                    model = load_model(model_path, model_type)
                    if model is None:
                        continue
                    
                    models_loaded = True
                    
                    # Process image for this specific model
                    processed_image = process_image(image, model_type)
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
                        
                        # Display results in card format
                        with cols[col_idx % 3]:
                            st.markdown(f"""
                            <div style='padding: 20px; border-radius: 10px; border: 1px solid #ddd; margin: 10px 0;'>
                                <h3>{model_type.upper()}</h3>
                                <p>Prediction: <strong>{prediction}</strong></p>
                                <p>Confidence: {confidence:.2%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        col_idx += 1
                    except Exception as e:
                        st.error(f"Error making prediction with {model_type}: {str(e)}")
                
                progress_bar.progress((i + 1) / len(model_files))
            
            if not models_loaded:
                st.error("No models could be loaded! Please check the model files.")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Error in main app: {str(e)}")

if __name__ == "__main__":
    main()