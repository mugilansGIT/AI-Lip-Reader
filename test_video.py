import cv2

video_path = (
    "data/raw_videos/"
    "s1_processed/video/mpg_6000/bbaf2n.mpg"
)

cap = cv2.VideoCapture(video_path)

count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    count += 1

print("Frames:", count)