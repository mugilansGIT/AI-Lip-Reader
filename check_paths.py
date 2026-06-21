import os

ROOT = "data/raw_videos"

for speaker in os.listdir(ROOT):

    speaker_path = os.path.join(ROOT, speaker)

    print("\nSpeaker:", speaker)

    for root, dirs, files in os.walk(speaker_path):

        print(root)

        mpg_files = [
            f for f in files
            if f.endswith(".mpg")
        ]

        if mpg_files:

            print("FOUND MPG FILES")

            for f in mpg_files[:5]:
                print(" ", f)

            break