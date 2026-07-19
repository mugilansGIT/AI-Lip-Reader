import cv2

from src.preprocessing.face_detector import LipDetector

detector = LipDetector()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("Press Q to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    lips = detector.extract_lips(frame)

    # Original webcam
    cv2.imshow("Webcam", frame)

    # Lip crop
    if lips is not None and lips.size > 0:
        cv2.imshow("Detected Lips", lips)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()