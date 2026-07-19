from src.preprocessing.webcam_capture import (
    capture_webcam_frames
)

frames = capture_webcam_frames()

print()

print("Shape :", frames.shape)

print("Done!")