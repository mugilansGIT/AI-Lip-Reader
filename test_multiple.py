from src.inference.sliding_predict import sliding_predict

videos = [
    "data/raw_videos_mp4/s1_processed/bbaf2n.mp4",
    "data/raw_videos_mp4/s1_processed/bbaf3s.mp4",
    "data/raw_videos_mp4/s1_processed/bbaf4p.mp4",
]

for video in videos:

    print("\n" + "=" * 60)
    print(video)

    subs = sliding_predict(video)

    for s in subs:
        print(s["text"])