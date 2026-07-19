import cv2

BUFFER_SIZE = 75

buffer = []

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("Press Q to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    # Add newest frame
    buffer.append(frame)

    # Keep only latest 75
    if len(buffer) > BUFFER_SIZE:
        buffer.pop(0)

    # Display frame count
    cv2.putText(
        frame,
        f"Frames: {len(buffer)}/{BUFFER_SIZE}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("AI Lip Reader", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()