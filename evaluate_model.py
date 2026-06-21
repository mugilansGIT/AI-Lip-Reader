import os
import random

from src.inference.sliding_predict import sliding_predict

# =====================================
# CONFIG
# =====================================

VIDEO_ROOT = "data/raw_videos_mp4"

NUM_SAMPLES = 20

# =====================================
# FIND ALL MP4 FILES
# =====================================

all_videos = []

for root, dirs, files in os.walk(VIDEO_ROOT):

    for file in files:

        if file.endswith(".mp4"):
            all_videos.append(
                os.path.join(root, file)
            )

print(f"Found {len(all_videos)} videos")

random.shuffle(all_videos)

test_videos = all_videos[:NUM_SAMPLES]

# =====================================
# READ ALIGNMENT
# =====================================

def read_alignment(path):

    words = []

    with open(path, "r") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 3:
                continue

            word = parts[2]

            if word == "sil":
                continue

            words.append(word)

    return " ".join(words)

# =====================================
# EVALUATION
# =====================================

total_words = 0
correct_words = 0

print("\n" + "=" * 80)
print("MODEL EVALUATION")
print("=" * 80)

for video_path in test_videos:

    filename = os.path.basename(video_path)

    speaker = os.path.basename(
        os.path.dirname(video_path)
    )

    # s1_processed -> s1_processed
    align_path = os.path.join(
        "data",
        "raw_videos",
        speaker,
        "align",
        filename.replace(".mp4", ".align")
    )

    if not os.path.exists(align_path):

        print("\nMissing:", align_path)
        continue

    ground_truth = read_alignment(
        align_path
    )

    try:

        subs = sliding_predict(video_path)

        if len(subs) > 0:
            prediction = subs[0]["text"]
        else:
            prediction = ""

    except Exception as e:

        print("ERROR:", filename)
        print(e)
        continue

    gt_words = ground_truth.split()
    pred_words = prediction.split()

    matches = sum(
        1
        for gt, pred in zip(
            gt_words,
            pred_words
        )
        if gt == pred
    )

    total_words += len(gt_words)
    correct_words += matches

    print("\n" + "-" * 80)
    print("VIDEO :", filename)
    print("GT    :", ground_truth)
    print("PRED  :", prediction)
    print(
        f"CORRECT: {matches}/{len(gt_words)}"
    )

# =====================================
# FINAL RESULT
# =====================================

print("\n" + "=" * 80)

if total_words > 0:

    accuracy = (
        correct_words
        / total_words
        * 100
    )

    print(
        f"WORD ACCURACY: "
        f"{accuracy:.2f}%"
    )

else:

    print("No samples evaluated.")

print("=" * 80)