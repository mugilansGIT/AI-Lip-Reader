from src.preprocessing.webcam_capture import capture_webcam_frames
from src.preprocessing.process_frames import process_frames
from src.inference.predict_core import predict_array


def webcam_predict():

    print("Look at the camera...")

    frames = capture_webcam_frames()

    print("Processing...")

    processed = process_frames(frames)

    if len(processed) == 0:
        return "No face detected."

    prediction = predict_array(processed)

    return prediction


if __name__ == "__main__":

    text = webcam_predict()

    print("\nPrediction:")
    print(text)