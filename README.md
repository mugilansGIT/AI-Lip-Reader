<p align="center">
  <img src="app/assets/banner.jpg" width="900"/>
</p>

<h1 align="center">🎥 AI Lip Reader</h1>

<p align="center">
  Real-Time Visual Speech Recognition using Deep Learning
</p>

# 🎯 AI Lip Reader

AI-powered lip reading system that converts silent videos into text transcripts and downloadable subtitle files (.srt).

## 🚀 Demo

📹 Full Project Demonstration

https://drive.google.com/file/d/1yTKMR-n57uhGuUfvYZJBQRt0HNGkx8q4/view?usp=sharing

---

## 📌 Features

* Upload silent videos through a Streamlit web interface
* Automatic face and lip detection using MediaPipe
* Lip sequence preprocessing pipeline
* Deep Learning based Lip Reading Model (LipNet-style architecture)
* Subtitle generation (.srt)
* GPU accelerated training with PyTorch
* Real-time prediction through a web application

---

## 🖼️ Screenshots

### Upload Video

![Upload Screen](screenshots/upload_screen.png)

### Prediction Output

![Prediction Output](screenshots/prediction_output.png.png)

### Subtitle Generation

![Subtitle Download](screenshots/srt_download.png)

---

## 🏗️ Project Architecture

Video Input

↓

Face Detection (MediaPipe)

↓

Lip Region Extraction

↓

Frame Preprocessing

↓

Lip Reading Model (PyTorch)

↓

CTC Decoding

↓

Text Prediction

↓

SRT Subtitle Generation

---

## 🧠 Tech Stack

### Machine Learning

* PyTorch
* NumPy

### Computer Vision

* OpenCV
* MediaPipe

### Web Application

* Streamlit

### Dataset

* GRID Corpus Dataset

---

## 📂 Project Structure

```text
lip-reader/
├── app/
├── data/
├── models/
├── src/
│   ├── preprocessing/
│   ├── model/
│   ├── inference/
│   └── utils/
├── screenshots/
├── requirements.txt
└── README.md
```

## 📊 Evaluation Results

Evaluated on 100 random GRID Corpus videos.

| Metric | Score |
|----------|----------|
| Word Accuracy | 90.31% |
| Word Error Rate (WER) | 5.58% |
| Character Error Rate (CER) | 2.36% |

Results generated using automated evaluation on unseen test samples.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/mugilansGIT/AI-Lip-Reader.git
cd AI-Lip-Reader
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app/app.py
```

Open:

```text
http://localhost:8501
```

Upload a video and generate subtitles.

---

## 🎓 Training Details

* Dataset: GRID Corpus
* Processed Samples: ~33,000
* Framework: PyTorch
* GPU: NVIDIA RTX 4060 Laptop GPU
* Loss Function: CTC Loss
* Decoder: Greedy CTC Decoder

---

## 🔮 Future Improvements

* Real-time webcam lip reading
* Beam Search Decoder
* Transformer-based lip reading models
* Speaker-independent evaluation
* Cloud deployment

---

## 👨‍💻 Author

Mugilan

Artificial Intelligence and Data Science Engineering Student

Interested in:

* Artificial Intelligence
* Computer Vision
* Data Science
* Machine Learning
