# Driver Drowsiness Detection System
# File: verify_environment.py
# Purpose: Verify that the required Python packages are working.

import sys


def check_python():
    print("Python version:")
    print(sys.version)
    print()


def check_numpy():
    try:
        import numpy as np

        print(f"[PASS] NumPy      : {np.__version__}")
        return True

    except Exception as e:
        print(f"[FAIL] NumPy      : {e}")
        return False


def check_opencv():
    try:
        import cv2

        print(f"[PASS] OpenCV     : {cv2.__version__}")
        return True

    except Exception as e:
        print(f"[FAIL] OpenCV     : {e}")
        return False


def check_mediapipe():
    try:
        import mediapipe as mp

        version = getattr(mp, "__version__", "version unavailable")

        print(f"[PASS] MediaPipe  : {version}")
        return True

    except Exception as e:
        print(f"[FAIL] MediaPipe  : {e}")
        return False


def main():
    print("=" * 55)
    print(" DRIVER DROWSINESS DETECTION")
    print(" ENVIRONMENT VERIFICATION")
    print("=" * 55)
    print()

    check_python()

    results = [
        check_numpy(),
        check_opencv(),
        check_mediapipe()
    ]

    print()
    print("=" * 55)

    if all(results):
        print("[SUCCESS] Environment verification passed.")
        print("The project is ready for the camera module.")

    else:
        print("[FAILED] Environment verification failed.")
        print("Check the package marked as [FAIL].")

    print("=" * 55)


if __name__ == "__main__":
    main()