import os
import glob

from src.preprocessing.frame_pipeline import process_video

speakers = glob.glob(
    "data/raw_videos/*_processed"
)

for speaker in speakers:

    videos = glob.glob(
        os.path.join(speaker, "*.mpg")
    )

    for video in videos:

        name = os.path.basename(video)

        out_dir = speaker.replace(
            "raw_videos",
            "processed"
        )

        os.makedirs(out_dir, exist_ok=True)

        output = os.path.join(
            out_dir,
            name.replace(".mpg", ".npy")
        )

        if os.path.exists(output):
            continue

        print("Processing:", video)

        process_video(video, output)