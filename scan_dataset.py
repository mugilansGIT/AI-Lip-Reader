import os

root = "data/raw_videos"

for speaker in os.listdir(root):

    speaker_path = os.path.join(root, speaker)

    if not os.path.isdir(speaker_path):
        continue

    print(f"\nSpeaker: {speaker}")

    for file in os.listdir(speaker_path):

        if file.endswith(".mpg"):
            print("Video:", file)

            align_file = file.replace(".mpg", ".align")

            align_path = os.path.join(
                speaker_path,
                "align",
                align_file
            )

            print(
                "Alignment exists:",
                os.path.exists(align_path)
            )

            break