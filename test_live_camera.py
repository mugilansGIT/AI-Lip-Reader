import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("Press Q to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("AI Lip Reader", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()