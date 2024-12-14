# import os
# import torch
# import mlflow
# from mlflow.pytorch import load_model
# from IPython.display import FileLink, display
# import shutil
# import yaml  # Import yaml for reading meta.yaml

# def get_model_name_from_metadata(model_dir):
#     """
#     Extracts the model name from the 'meta.yaml' file located in the grandparent directory of the given model directory.
    
#     Args:
#     - model_dir (str): Path to the MLflow model directory.
    
#     Returns:
#     - model_name (str): Extracted model name or a default name.
#     """
#     # Navigate to the grandparent directory of the model_dir
#     grandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(model_dir)))
    
#     # Construct the path to 'meta.yaml' in the grandparent directory
#     meta_file = os.path.join(grandparent_dir, "meta.yaml")
    
#     # Check if the 'meta.yaml' exists in the grandparent directory
#     if os.path.exists(meta_file):
#         try:
#             with open(meta_file, "r") as f:
#                 meta_data = yaml.safe_load(f)  # Load the YAML file
#                 # Extract the 'name' attribute from the YAML data
#                 if 'name' in meta_data:
#                     model_name = meta_data['name']
#                     return model_name
#                 else:
#                     print(f"No 'name' found in {meta_file}. Using 'unknown_model'.")
#                     return "unknown_model"
#         except Exception as e:
#             print(f"Error reading {meta_file}: {e}")
#             return "unknown_model"


# def check_model_integrity(model_file):
#     """
#     Verifies if the saved PyTorch model can be loaded successfully.
    
#     Args:
#     - model_file (str): Path to the saved PyTorch model.
    
#     Returns:
#     - bool: True if the model loads successfully, False otherwise.
#     """
#     try:
#         # Attempt to load the state_dict into a dummy model structure
#         state_dict = torch.load(model_file, map_location=torch.device('cpu'))
#         if isinstance(state_dict, dict):  # Check if it's a valid state_dict
#             print(f"Model {model_file} successfully verified!")
#             return True
#     except Exception as e:
#         print(f"Failed to verify model {model_file}: {e}")
#     return False


# def convert_and_save_models_with_metadata(input_dir, output_dir):
#     """
#     Converts and saves models to PyTorch format using MLflow, with naming based on metadata.
    
#     Args:
#     - input_dir (str): Path to the input directory containing MLflow models.
#     - output_dir (str): Path to the directory where converted models will be saved.
#     """
#     # Ensure the output directory exists
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Walk through the directory to find MLflow model files
#     for root, dirs, files in os.walk(input_dir):
#         if "MLmodel" in files:  # Identifies MLflow model directories
#             try:
#                 print(f"Processing MLflow model in: {root}")
                
#                 # Load the model using MLflow
#                 mlflow_model = load_model(root)
                
#                 # Extract the model name from metadata
#                 model_name = get_model_name_from_metadata(root)
#                 model_file = os.path.join(output_dir, f"{model_name}_converted.pth")
                
#                 # Save the PyTorch state_dict for portability
#                 torch.save(mlflow_model.state_dict(), model_file)
                
#                 # Check if the model was saved correctly
#                 if check_model_integrity(model_file):
#                     print(f"Model saved and verified: {model_file}")
#                 else:
#                     print(f"Model failed verification: {model_file}")
            
#             except Exception as e:
#                 print(f"Failed to process {root}: {e}")


# def zip_converted_models(output_dir):
#     """
#     Zips the output directory containing converted models for download.
    
#     Args:
#     - output_dir (str): Path to the directory containing converted models.
    
#     Returns:
#     - str: Path to the zipped file.
#     """
#     zip_path = f"{output_dir}.zip"
#     shutil.make_archive(output_dir, 'zip', output_dir)
#     print(f"Zipped models folder: {zip_path}")
#     return zip_path


# # Set paths
# input_dir = "/kaggle/input/ddp-v4-run-1"  # Root directory containing models
# output_dir = "/kaggle/working/converted_models"  # Output directory for converted models

# # Convert and save models
# convert_and_save_models_with_metadata(input_dir, output_dir)

# # Zip the converted models folder
# zip_path = zip_converted_models(output_dir)

# # Create download links for converted models
# print("\nDownload links for converted models:")
# for file in os.listdir(output_dir):
#     file_path = os.path.join(output_dir, file)
#     display(FileLink(file_path))

# # Provide download link for the zipped models folder
# print("\nDownload link for zipped models folder:")
# display(FileLink(zip_path))