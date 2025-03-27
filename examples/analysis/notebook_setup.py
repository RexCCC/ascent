import os
import sys

def setup_notebook():
    # Set the working directory to the ascent root
    ascent_path = os.path.join(os.getcwd().split('ascent')[0], 'ascent')
    os.chdir(ascent_path)
    
    # Add ascent to sys.path
    if ascent_path not in sys.path:
        sys.path.append(ascent_path)

    # Verify setup
    print(f"Working directory set to: {os.getcwd()}")
    print(f"'ascent' folder added to sys.path: {ascent_path}")