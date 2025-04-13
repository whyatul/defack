import os
import sys
import json
import shutil
from pathlib import Path

def setup_kaggle_credentials():
    """Set up Kaggle API credentials."""
    print("Setting up Kaggle credentials...")
    
    # Check if credentials already exist
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if kaggle_json.exists():
        print("Kaggle credentials already exist.")
        return True
    
    # Create directory if it doesn't exist
    kaggle_dir.mkdir(exist_ok=True)
    
    # Get credentials from user
    print("\nYou need to set up your Kaggle credentials.")
    print("1. Go to kaggle.com and sign in")
    print("2. Click on your profile picture → Account")
    print("3. Scroll down to the API section")
    print("4. Click 'Create New API Token' and download kaggle.json")
    
    kaggle_json_path = input("\nEnter the path to your downloaded kaggle.json file: ")
    
    try:
        # Copy credentials to ~/.kaggle/
        shutil.copy(kaggle_json_path, kaggle_json)
        os.chmod(kaggle_json, 0o600)
        print("Kaggle credentials configured successfully!")
        return True
    except FileNotFoundError:
        print(f"Error: Could not find file {kaggle_json_path}")
        return False
    except Exception as e:
        print(f"Error setting up Kaggle credentials: {e}")
        return False

def download_models():
    """Download the required model files using Kaggle API."""
    # Check if models are already downloaded
    model_files = [
        "cnn_transformer_converted.pth",
        "cross_attention_converted.pth",
        "efficientnet_converted.pth",
        "swin_converted.pth",
        "two_stream_converted.pth", 
        "xception_converted.pth"
    ]
    
    # Check if all model files already exist
    all_exist = True
    for model_file in model_files:
        if not Path(model_file).exists():
            all_exist = False
            break
    
    if all_exist:
        print("Model files already exist, skipping download.")
        return True
    
    try:
        import kaggle
        print("\nDownloading model files...")
        # Update to use the correct dataset ID based on your output
        kaggle.api.dataset_download_files('ameencaslam/ddp-v4-models', path='.')
        print("Extracting model files...")
        os.system("unzip -o ddp-v4-models.zip")
        os.system("mv ddp-v4-models/* .")
        print("Model files downloaded and extracted successfully!")
        return True
    except Exception as e:
        print(f"Error downloading models: {e}")
        return False

def fix_torch_streamlit_issue():
    """Fix the torch/streamlit integration issue by running the fix script."""
    try:
        print("\nFixing PyTorch/Streamlit integration issue...")
        # Run the permanent fix from our patch script
        fix_script = Path(__file__).parent / "fix_streamlit_torch.py"
        if not fix_script.exists():
            print(f"Warning: Fix script not found at {fix_script}. Cannot apply fix.")
            return False
        
        # Run the fix script as a module
        print(f"Running fix script: {fix_script}")
        result = os.system(f"{sys.executable} {fix_script}")
        
        if result == 0:
            print("Successfully applied PyTorch/Streamlit fix.")
            return True
        else:
            print(f"Error applying PyTorch/Streamlit fix. Exit code: {result}")
            return False
    except Exception as e:
        print(f"Error fixing PyTorch/Streamlit issue: {e}")
        return False

if __name__ == "__main__":
    print("Setting up deepfake detection environment...")
    
    # Setup steps
    kaggle_setup = setup_kaggle_credentials()
    if kaggle_setup:
        models_downloaded = download_models()
    else:
        print("Skipping model download due to credential setup failure.")
        models_downloaded = False
    
    # Apply the PyTorch/Streamlit fix
    streamlit_fixed = fix_torch_streamlit_issue()
    
    # Summary
    print("\n=== Setup Summary ===")
    print(f"Kaggle credentials: {'✓ Configured' if kaggle_setup else '✗ Failed'}")
    print(f"Model files: {'✓ Downloaded' if models_downloaded else '✗ Not downloaded'}")
    print(f"PyTorch/Streamlit fix: {'✓ Applied' if streamlit_fixed else '✗ Failed'}")
    
    if not all([kaggle_setup, models_downloaded, streamlit_fixed]):
        print("\nSome setup steps failed. Please check the errors above.")
        if not streamlit_fixed:
            print("\nTIP: To fix the PyTorch/Streamlit issue manually, run:")
            print(f"python {Path(__file__).parent / 'fix_streamlit_torch.py'}")
    else:
        print("\nSetup completed successfully! You can now run your Streamlit app.")
