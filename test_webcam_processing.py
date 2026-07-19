from src.preprocessing.webcam_capture import capture_webcam_frames
from src.preprocessing.process_frames import process_frames

frames = capture_webcam_frames()

processed = process_frames(frames)

print()

print("Processed Shape :", processed.shape)

print("Min :", processed.min())

print("Max :", processed.max())