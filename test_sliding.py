from src.inference.sliding_predict import sliding_predict


video_path = (
    "data/raw_videos_mp4/"
    "s1_processed/bbaf2n.mp4"
)


subs = sliding_predict(video_path)

print("\nFinal Subtitles:\n")

for s in subs:

    print(
        f"{s['start']:.2f}s "
        f"--> "
        f"{s['end']:.2f}s"
    )

    print(s["text"])

    print("-" * 40)