import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            ".."
        )
    )
)

import cv2
import numpy as np
import threading

from src.preprocessing.process_frames import process_frames
from src.inference.predict_core import predict_array

# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# ==========================================
# CONFIG
# ==========================================

BUFFER_SIZE = 75
PREDICT_EVERY = 25

buffer = []
frame_counter = 0

current_prediction = "Waiting for prediction..."
predicting = False

print("Press Q to quit")

# ==========================================
# BACKGROUND PREDICTION
# ==========================================

def prediction_worker(frames):

    global current_prediction
    global predicting

    try:

        processed = process_frames(
            np.array(frames)
        )

        if len(processed) > 0:

            prediction = predict_array(
                processed
            )

            current_prediction = prediction

            print("Prediction:", prediction)

        else:

            current_prediction = "No face detected"

    except Exception as e:

        print(e)
        current_prediction = "Prediction Error"

    predicting = False


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    # Add newest frame
    buffer.append(frame)

    if len(buffer) > BUFFER_SIZE:
        buffer.pop(0)

    frame_counter += 1

    # ======================================
    # Launch prediction thread
    # ======================================

    if (
        len(buffer) == BUFFER_SIZE
        and
        frame_counter % PREDICT_EVERY == 0
        and
        not predicting
    ):

        predicting = True

        threading.Thread(
            target=prediction_worker,
            args=(buffer.copy(),),
            daemon=True
        ).start()

    # ======================================
    # Draw subtitle background
    # ======================================

    h, w = frame.shape[:2]

    cv2.rectangle(
        frame,
        (0, h - 60),
        (w, h),
        (0, 0, 0),
        -1
    )

    # ======================================
    # Subtitle
    # ======================================

    cv2.putText(
        frame,
        current_prediction,
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # ======================================
    # Frame Counter
    # ======================================

    cv2.putText(
        frame,
        f"Frames: {len(buffer)}/{BUFFER_SIZE}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # ======================================
    # Status
    # ======================================

    if len(buffer) < BUFFER_SIZE:

        status = "Buffering..."

    elif predicting:

        status = "Predicting..."

    else:

        status = "AI Listening..."

    cv2.putText(
        frame,
        status,
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # ======================================
    # Display
    # ======================================

    cv2.imshow(
        "AI Lip Reader - Live Webcam",
        frame
    )

    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()