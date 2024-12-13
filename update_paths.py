import os
import yaml

def update_yaml_paths():
    models_dir = os.path.join("mlruns", "models")
    current_dir = os.getcwd()  # Get current working directory
    
    # Walk through all directories in mlruns
    for model_name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_name)
        if os.path.isdir(model_path):
            # Process each version folder
            versions = [d for d in os.listdir(model_path) if d.startswith('version-')]
            for version in versions:
                version_path = os.path.join(model_path, version)
                meta_yaml_path = os.path.join(version_path, "meta.yaml")
                
                if os.path.exists(meta_yaml_path):
                    # Read the YAML file
                    with open(meta_yaml_path, 'r') as f:
                        data = yaml.safe_load(f)
                    
                    # Get the run ID and artifacts path from the old source path
                    if data.get('source', '').startswith('file:///kaggle/working/'):
                        old_path = data['source'].replace('file:///kaggle/working/deepfake-detection-project-v4/', '')
                        run_id = old_path.split('/')[1]
                        
                        # Create new local path
                        new_path = f"file:///{current_dir.replace(os.sep, '/')}/mlruns/{run_id}/artifacts/best_model"
                        
                        # Update paths
                        data['source'] = new_path
                        data['storage_location'] = new_path
                        
                        # Save the updated YAML
                        with open(meta_yaml_path, 'w') as f:
                            yaml.safe_dump(data, f)
                        
                        print(f"Updated {meta_yaml_path}")

if __name__ == "__main__":
    update_yaml_paths()