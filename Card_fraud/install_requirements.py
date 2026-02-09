import subprocess
import sys
import os

# Folders we want to ensure exist
PROJECT_FOLDERS = [
    "data",
    "src",
    "docker",
    "artifacts",
    "notebooks",
    "scripts"
]

def create_folders():
    for folder in PROJECT_FOLDERS:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"[INFO] Created folder: {folder}")
        else:
            print(f"[INFO] Folder already exists: {folder}")

def install_requirements(requirements_path="docker/requirements.txt"):
    if not os.path.exists(requirements_path):
        print(f"[ERROR] Requirements file not found at {requirements_path}")
        sys.exit(1)

    print(f"[INFO] Installing packages from {requirements_path}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
    print("[SUCCESS] All requirements installed.")

def verify_install():
    try:
        import numpy, torch, tensorflow as tf
        print(f"numpy: {numpy.__version__}")
        print(f"torch: {torch.__version__}")
        print(f"tensorflow: {tf.__version__}")
    except Exception as e:
        print("[ERROR] Verification failed:", e)

if __name__ == "__main__":
    print("[INFO] Setting up project environment...")
    create_folders()
    install_requirements()
    verify_install()
    print("[DONE] Environment + folders ready.")
