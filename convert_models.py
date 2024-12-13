import torch
import os
import glob
import logging
import shutil
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_model_type(run_dir):
    """Determine model type from MLflow metadata"""
    try:
        # Try to read meta.yaml
        meta_path = os.path.join(run_dir, "meta.yaml")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                content = f.read().lower()
                if "xception" in content:
                    return "xception"
                elif "efficientnet" in content:
                    return "efficientnet"
                elif "swin" in content:
                    return "swin"
                elif "cross_attention" in content:
                    return "cross_attention"
                elif "two_stream" in content:
                    return "two_stream"
                elif "cnn_transformer" in content:
                    return "cnn_transformer"
        
        # Try to read from parent directory name
        parent_dir = os.path.basename(os.path.dirname(run_dir)).lower()
        if "xception" in parent_dir:
            return "xception"
        elif "efficientnet" in parent_dir:
            return "efficientnet"
        elif "swin" in parent_dir:
            return "swin"
        elif "cross_attention" in parent_dir:
            return "cross_attention"
        elif "two_stream" in parent_dir:
            return "two_stream"
        elif "cnn_transformer" in parent_dir:
            return "cnn_transformer"
            
    except Exception as e:
        logger.error(f"Error reading metadata: {str(e)}")
    return None

def convert_model(run_dir, output_dir):
    """Convert MLflow model to simple PyTorch state dict"""
    try:
        # Find the model file
        model_files = glob.glob(os.path.join(run_dir, "**", "model.pth"), recursive=True)
        if not model_files:
            logger.warning(f"No model.pth found in {run_dir}")
            return False
        
        model_path = model_files[0]
        logger.info(f"Found model at: {model_path}")
        
        # Determine model type
        model_type = get_model_type(run_dir)
        if model_type is None:
            logger.warning(f"Could not determine model type for {run_dir}")
            return False
            
        logger.info(f"Detected model type: {model_type}")
        
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
            return False
        
        if model is None:
            logger.warning(f"Could not initialize model of type {model_type}")
            return False
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{model_type}_converted.pth")
        
        # Save initialized model state dict
        torch.save(model.state_dict(), output_path, _use_new_zipfile_serialization=True)
        logger.info(f"Saved converted model to {output_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error converting model: {str(e)}")
        return False

def main():
    # MLflow runs directory
    mlruns_dir = "mlruns"
    if not os.path.exists(mlruns_dir):
        logger.error(f"mlruns directory not found at {os.path.abspath(mlruns_dir)}!")
        return
    
    # Output directory for converted models
    output_dir = "converted_models"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all experiment directories
    experiment_dirs = [d for d in glob.glob(os.path.join(mlruns_dir, "*")) 
                      if os.path.isdir(d) and not d.endswith('models')]
    
    if not experiment_dirs:
        logger.error("No experiment directories found!")
        return
    
    logger.info(f"Found {len(experiment_dirs)} experiment directories")
    
    # Convert each model
    success_count = 0
    for exp_dir in experiment_dirs:
        run_dirs = glob.glob(os.path.join(exp_dir, "*"))
        for run_dir in run_dirs:
            if os.path.isdir(run_dir):
                if convert_model(run_dir, output_dir):
                    success_count += 1
    
    logger.info(f"Successfully converted {success_count} models")

if __name__ == "__main__":
    main() 