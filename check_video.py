import cv2

video_path = r"data/raw_videos_mp4/s1_processed/bbaf2n.mp4"

cap = cv2.VideoCapture(video_path)

print("Opened:", cap.isOpened())

ret, frame = cap.read()

print("First frame read:", ret)

if ret:
    print("Shape:", frame.shape)

cap.release()