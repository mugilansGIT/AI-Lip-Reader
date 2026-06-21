from src.inference.sliding_predict import sliding_predict
from src.inference.subtitle_writer import write_srt

video_path = (
    "data/raw_videos_mp4/"
    "s1_processed/bbaf2n.mp4"
)

subs = sliding_predict(video_path)

write_srt(subs, "output.srt")

print("Done.")