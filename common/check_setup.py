"""Quick check that Python, the libraries and the webcam all work.

Run with:  python main.py check
"""
import sys


def version_of(module_name):
    """The library's version, or a note if it isn't installed."""
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", "installed")
    except ImportError:
        return "NOT INSTALLED (run: pip install -r requirements.txt)"


def main():
    print(f"Python       : {sys.version.split()[0]}")
    for name, module_name in [("OpenCV", "cv2"), ("MediaPipe", "mediapipe"), ("scikit-learn", "sklearn"),
                              ("PyTorch", "torch"), ("Transformers", "transformers")]:
        print(f"{name:13s}: {version_of(module_name)}")

    # Open the webcam exactly like the apps do, and grab one frame.
    from common.camera import Camera
    try:
        camera = Camera()
    except RuntimeError as error:
        print(f"webcam FAILED: {error}")
        return
    frame = next(camera.frames(), None)
    camera.release()

    if frame is not None:
        height, width = frame.shape[:2]
        print(f"webcam OK ({width}x{height})")
    else:
        print("webcam FAILED: check that no other app (Teams, Zoom, Camera) is using it,")
        print("and that camera access is allowed in Windows Settings > Privacy & security > Camera.")


if __name__ == "__main__":
    main()
