from src.preprocessing.frame_pipeline import process_video
import numpy as np

video_path = "data/raw_videos_mp4/s1_processed/bbaf2n.mp4"
output_path = "test.npy"

process_video(video_path, output_path)

data = np.load(output_path)

print("Shape:", data.shape)