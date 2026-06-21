from src.utils.video_utils import read_video
from src.preprocessing.face_detector import LipDetector

video_path = "data/raw_videos_mp4/s1_processed/bbaf2n.mp4"

frames = read_video(video_path)

print("Video frames:", len(frames))

detector = LipDetector()

success = 0

for frame in frames:

    lips = detector.extract_lips(frame)

    if lips is not None:
        success += 1

print("Lip detections:", success)