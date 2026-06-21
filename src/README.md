# AI Lip Reader

## Overview

AI-based lip reading system that converts silent videos into subtitles using deep learning.

## Features

* Lip detection using MediaPipe
* Video preprocessing with OpenCV
* LipNet-based neural network
* Automatic subtitle generation
* SRT subtitle export
* Streamlit web interface

## Dataset

GRID Corpus

33,000+ processed training samples

## Technologies Used

* Python
* PyTorch
* OpenCV
* MediaPipe
* Streamlit

## Results

* Final Training Loss: 0.0626
* Approximate Word Accuracy: 95%+

## How to Run

1. Install dependencies:

pip install -r requirements.txt

2. Run application:

streamlit run app/app.py

## Future Improvements

* Real-time webcam lip reading
* Beam search decoding
* Multi-speaker evaluation
