
"""Check the local environment required for ASL sign-language recognition."""

import importlib
import importlib.metadata
import platform
import sys


PACKAGES = {
    "numpy": "NumPy",
    "cv2": "OpenCV",
    "torch": "PyTorch",
    "torchvision": "Torchvision",
    "mediapipe": "MediaPipe",
}


def main():
    print("=" * 60)
    print("ASL SIGN LANGUAGE RECOGNITION - ENVIRONMENT CHECK")
    print("=" * 60)

    print(f"Python version : {platform.python_version()}")
    print(f"Python path    : {sys.executable}")
    print(f"Operating system: {platform.platform()}")
    print()

    for module_name, display_name in PACKAGES.items():
        try:
            importlib.import_module(module_name)

            try:
                version = importlib.metadata.version(
                    "opencv-python" if module_name == "cv2"
                    else module_name
                )
            except importlib.metadata.PackageNotFoundError:
                version = "installed (version unavailable)"

            print(f"[OK]   {display_name}: {version}")

        except Exception as exc:
            print(f"[FAIL] {display_name}: {type(exc).__name__}: {exc}")

    print()
    try:
        import torch

        print(f"PyTorch CUDA build: {torch.version.cuda}")
        print(f"CUDA available    : {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"GPU               : {torch.cuda.get_device_name(0)}")
        else:
            print("GPU               : CPU mode will be required")
    except Exception as exc:
        print(f"CUDA check failed: {type(exc).__name__}: {exc}")

    print()
    print("Environment check finished.")
    print("Review any [FAIL] lines before running preprocessing or training.")


if __name__ == "__main__":
    main()
