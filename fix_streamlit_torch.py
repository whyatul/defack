import importlib
import types
import os
import sys
from pathlib import Path
import fileinput
import re

def apply_runtime_patch():
    """Apply a runtime patch to fix Streamlit and PyTorch integration issue"""
    try:
        # Import the problematic module
        import streamlit.watcher.local_sources_watcher as lsw
        
        # Define a safer version of extract_paths
        def safe_extract_paths(module):
            try:
                if hasattr(module, '__path__') and isinstance(module.__path__, types.ModuleType):
                    return []
                elif hasattr(module, '__path__') and hasattr(module.__path__, '_path'):
                    return list(module.__path__._path)
                elif hasattr(module, '__path__'):
                    return module.__path__
                return []
            except (RuntimeError, AttributeError):
                return []
        
        # Replace the function
        lsw.extract_paths = safe_extract_paths
        print("✓ Applied runtime patch for Streamlit/PyTorch integration issue")
        return True
    except Exception as e:
        print(f"✗ Failed to apply runtime patch: {e}")
        return False

def apply_permanent_patch():
    """Permanently fix the local_sources_watcher.py file"""
    try:
        # Find the streamlit package location
        import streamlit
        streamlit_path = Path(streamlit.__file__).parent
        file_path = streamlit_path / "watcher" / "local_sources_watcher.py"
        
        if not file_path.exists():
            print(f"✗ Could not find file at {file_path}")
            return False
        
        # Create backup
        backup_path = str(file_path) + ".bak"
        if not Path(backup_path).exists():
            print(f"Creating backup of original file at {backup_path}")
            with open(file_path, 'r') as source:
                with open(backup_path, 'w') as backup:
                    backup.write(source.read())
        
        # Read the file content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Replace the problematic lambda function
        original_pattern = r"lambda m: list\(m\.__path__\._path\)"
        replacement = "lambda m: safe_extract_paths(m)"
        
        if original_pattern not in content:
            print("✗ Could not find the problematic code pattern in the file")
            return False
        
        # Add the safe_extract_paths function and replace the lambda
        safe_function = """
def safe_extract_paths(module):
    try:
        if not hasattr(module, '__path__'):
            return []
        if isinstance(module.__path__, types.ModuleType):
            return []
        elif hasattr(module.__path__, '_path'):
            return list(module.__path__._path)
        else:
            return module.__path__
    except (RuntimeError, AttributeError, TypeError):
        return []
"""
        
        # Add import for types if not present
        if "import types" not in content:
            content = content.replace("import sys", "import sys\nimport types")
        
        # Add the safe function if not already there
        if "def safe_extract_paths" not in content:
            # Find a good insertion point - after imports but before functions
            insert_point = content.find("def get_module_paths")
            if insert_point != -1:
                content = content[:insert_point] + safe_function + "\n" + content[insert_point:]
        
        # Replace the lambda
        modified_content = re.sub(original_pattern, replacement, content)
        
        # Write the modified content back
        with open(file_path, 'w') as f:
            f.write(modified_content)
        
        print(f"✓ Successfully patched {file_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to apply permanent patch: {e}")
        print(f"Error details: {str(e)}")
        return False

if __name__ == "__main__":
    print("Applying Streamlit/PyTorch compatibility fixes...")
    
    # Try both patches
    permanent_result = apply_permanent_patch()
    runtime_result = apply_runtime_patch()
    
    if permanent_result:
        print("\n✓ Applied permanent fix to Streamlit code.")
        print("✓ This fix will persist between Streamlit runs.")
        print("✓ You can now run your Streamlit app normally.")
    elif runtime_result:
        print("\n⚠️ Applied runtime fix only.")
        print("⚠️ You will need to run this script before each Streamlit session.")
        print("⚠️ Run: python fix_streamlit_torch.py")
    else:
        print("\n✗ Failed to apply fixes.")
        print("✗ You may continue to see PyTorch/Streamlit integration errors.")
