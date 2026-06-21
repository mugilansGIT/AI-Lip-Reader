import os
import subprocess


INPUT_ROOT = "data/raw_videos"
OUTPUT_ROOT = "data/raw_videos_mp4"


for root, dirs, files in os.walk(INPUT_ROOT):

    for file in files:

        if not file.endswith(".mpg"):
            continue

        input_file = os.path.join(
            root,
            file
        )

        relative_path = os.path.relpath(
            root,
            INPUT_ROOT
        )

        output_dir = os.path.join(
            OUTPUT_ROOT,
            relative_path
        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        output_file = os.path.join(
            output_dir,
            file.replace(".mpg", ".mp4")
        )

        print(f"Converting: {input_file}")

        command = [
            "ffmpeg",
            "-y",
            "-i",
            input_file,
            output_file
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:

            print("FFMPEG ERROR:")
            print(result.stderr)

        else:

            print(f"Saved: {output_file}")


print("\nConversion complete.")