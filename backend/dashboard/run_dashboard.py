import os
import subprocess
import sys

def main():
    """
    Launch the Streamlit dashboard from the correct directory.
    This ensures relative imports in the dashboard work correctly.
    """
    # Get the directory of this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the parent directory (project root)
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    # Add project root to Python path
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    # Set PYTHONPATH environment variable
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    if project_root not in current_pythonpath:
        os.environ["PYTHONPATH"] = f"{project_root}{os.pathsep}{current_pythonpath}" if current_pythonpath else project_root
    
    # Change to project root directory to ensure imports work
    os.chdir(project_root)
    
    # Print information
    print("Starting AlgoTrader Dashboard")
    print(f"Project Root: {project_root}")
    print(f"Python Path: {os.environ['PYTHONPATH']}")
    print("Dashboard URL: http://localhost:8501\n")
    
    # Start Streamlit with the dashboard home page
    dashboard_path = os.path.join("backend", "dashboard", "Home.py")
    
    # Create new environment with updated PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = os.environ["PYTHONPATH"]
    
    # Run Streamlit with the updated environment
    subprocess.run(["streamlit", "run", dashboard_path], env=env)

if __name__ == "__main__":
    main() 