import cv2
import numpy as np

from src.preprocessing.face_detector import LipDetector
from src.preprocessing.lip_extractor import preprocess_lips

detector = LipDetector()

cap = cv2.VideoCapture(0)

buffer = []

print("Press SPACE to capture 75 frames")
print("Press Q to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    lips = detector.extract_lips(frame)

    display = frame.copy()

    if lips is not None:

        processed = preprocess_lips(lips)

        cv2.imshow("Detected Lips", lips)

    cv2.imshow("Webcam", display)

    key = cv2.waitKey(1)

    if key == ord(" "):

        print("\nCapturing 75 frames...")

        buffer = []

        while len(buffer) < 75:

            ret, frame = cap.read()

            if not ret:
                continue

            lips = detector.extract_lips(frame)

            if lips is None:
                continue

            processed = preprocess_lips(lips)

            buffer.append(processed)

            cv2.imshow("Detected Lips", lips)
            cv2.imshow("Webcam", frame)

            cv2.waitKey(1)

            print(
                f"\rFrames: {len(buffer)}/75",
                end=""
            )

        print("\nCapture Complete!")

        data = np.array(buffer)

        print("Shape:", data.shape)
        print("Min :", data.min())
        print("Max :", data.max())

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()