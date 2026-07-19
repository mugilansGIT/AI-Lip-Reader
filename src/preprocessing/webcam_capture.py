import cv2
import numpy as np


def capture_webcam_frames(
    num_frames=75,
    camera_index=0
):
    """
    Capture a fixed number of frames
    from the webcam.

    Returns
    -------
    numpy.ndarray
        Shape: (T, H, W, 3)
    """

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(
            "Cannot open webcam."
        )

    frames = []

    print(f"Capturing {num_frames} frames...")

    while len(frames) < num_frames:

        ret, frame = cap.read()

        if not ret:
            continue

        frames.append(frame)

        cv2.imshow(
            "Press Q to Cancel",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    return np.array(frames)