# Lip Reader

An end-to-end lip reading system that takes a video as input and outputs a text transcript (and optional `.srt` subtitle file).

---

## Project Structure

```
├── data/
│   ├── raw_videos/       # Input videos
│   ├── processed/        # Cropped lip sequences
│   └── alignments/       # Label files (for training)
├── models/
│   ├── face_detector/    # Pre-trained face/landmark model files
│   └── lip_reader/       # Trained model weights
├── src/
│   ├── preprocessing/    # Face detection, lip cropping, pipeline
│   ├── model/            # Architecture, training loop, evaluation
│   ├── inference/        # Prediction & subtitle writing
│   └── utils/            # Video I/O, CTC decoding, text helpers
├── app/
│   ├── app.py            # Streamlit web UI
│   └── templates/        # HTML templates (Flask fallback)
├── notebooks/
│   └── explore.ipynb     # Experimentation notebook
├── config.yaml
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download pre-trained models

Place your face landmark model (e.g. `shape_predictor_68_face_landmarks.dat`) in `models/face_detector/`.

### 3. Preprocess data

```python
from src.preprocessing.frame_pipeline import FramePipeline
# See src/preprocessing/ for details
```

### 4. Train

```bash
python -m src.model.train
```

### 5. Run inference

```python
from src.inference.predict import LipReadingPredictor
predictor = LipReadingPredictor("models/lip_reader/best_model.pt", vocab, config)
print(predictor.predict("data/raw_videos/sample.mp4"))
```

### 6. Launch the web UI

```bash
streamlit run app/app.py
```

---

## Configuration

All paths and hyperparameters live in `config.yaml`. Edit that file to change model size, learning rate, data splits, etc.

---

## Metrics

| Metric | Description |
|--------|-------------|
| WER    | Word Error Rate  |
| CER    | Character Error Rate |

Computed in `src/model/evaluate.py` using edit distance.
