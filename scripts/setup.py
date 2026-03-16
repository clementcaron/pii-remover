#!/usr/bin/env python3
import subprocess
import os
import sys
import argparse

def check_nvidia_gpu():
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def get_python_exec():
    # If we are already running inside a venv, use its executable
    if sys.prefix != sys.base_prefix:
        return sys.executable
    # Otherwise check for .venv
    venv_python = os.path.join(".venv", "bin", "python")
    if os.path.exists(venv_python):
        return venv_python
    return sys.executable

def get_install_base():
    python_exec = get_python_exec()
    try:
        subprocess.check_output(["uv", "--version"], stderr=subprocess.STDOUT)
        return ["uv", "pip", "install", "--python", python_exec]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [python_exec, "-m", "pip", "install"]

def install_torch(gpu_available: bool):
    install_cmd = get_install_base()
    print(f"Installing torch (GPU Support: {gpu_available}) using {' '.join(install_cmd)}...")
    
    if gpu_available:
        # Install torch with CUDA support
        cmd_torch = install_cmd + ["torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu121"]
    else:
        # Install CPU torch
        cmd_torch = install_cmd + ["torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"]
    
    try:
        subprocess.check_call(cmd_torch)
        print("Torch installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install torch: {e}")
        sys.exit(1)

def download_model(model_name: str = "numind/NuNER_Zero"):
    install_cmd = get_install_base()
    python_exec = get_python_exec()
    print(f"Pre-downloading model: {model_name}...")
    try:
        # First ensure all requirements are installed
        print("Ensuring requirements from requirements.txt are installed...")
        subprocess.check_call(install_cmd + ["-r", "requirements.txt"])
        
        # Install huggingface-hub if not present
        subprocess.check_call(install_cmd + ["huggingface-hub"])
        
        # We don't download here explicitly to avoid redundancy with the app startup download
        # but we ensure the environment is ready.
        print(f"Environment ready for {model_name}.")
    except Exception as e:
        print(f"Failed to install requirements: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="PII Remover Proxy Setup Script")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU installation even if GPU is detected")
    parser.add_argument("--silent", action="store_true", help="Run without interactive confirmation (useful for scripts)")
    args = parser.parse_args()

    # Try to read USE_GPU from .env if it exists
    use_gpu_env = False
    if os.path.exists(".env"):
        try:
            with open(".env", "r") as f:
                for line in f:
                    if line.startswith("USE_GPU="):
                        use_gpu_env = line.split("=")[1].strip().lower() == "true"
                        break
        except Exception:
            pass

    gpu_exists = check_nvidia_gpu() and not args.force_cpu
    
    # Logic: 
    # 1. If --force-cpu is passed, use CPU.
    # 2. If USE_GPU=true in .env but no GPU is detected, warn and use CPU.
    # 3. If GPU is detected and USE_GPU is true (or we are in auto mode), use GPU.
    
    final_gpu = False
    if args.force_cpu:
        print("Forcing CPU setup as requested by flag.")
        final_gpu = False
    elif gpu_exists:
        if use_gpu_env:
            print("NVIDIA GPU detected and USE_GPU=true in .env. Proceeding with GPU setup.")
            final_gpu = True
        elif not args.silent:
            # Interactive prompt if not silent and no env preference
            ans = input("NVIDIA GPU detected. Enable GPU acceleration? (y/n): ").lower()
            final_gpu = ans == 'y'
        else:
            print("NVIDIA GPU detected. Defaulting to GPU setup in silent mode.")
            final_gpu = True
    else:
        if use_gpu_env:
            print("WARNING: USE_GPU=true in .env but no compatible NVIDIA GPU was detected. Falling back to CPU.")
        else:
            print("No NVIDIA GPU detected. Proceeding with CPU setup.")
        final_gpu = False

    install_torch(final_gpu)
    download_model()

    print("\nSetup complete!")

if __name__ == "__main__":
    main()
