# 👤 FaceVision AI - Pure Python Face Detection Application

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/CV-OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **FaceVision AI** is a pure Python-powered computer vision web application for detecting human faces in real-time. Built using **Streamlit** and **OpenCV**, it features adaptive contrast illumination balancing (CLAHE), multi-scale cascade merging, webcam capture, local image upload, and an isolated face crops inspector gallery.

---

## 🌟 Key Features

- 🐍 **100% Pure Python**: Fully written in Python using Streamlit and OpenCV.
- 📁 **Local Image Upload**: Directly upload images (PNG, JPG, WebP) from your local computer storage.
- 📸 **Live Webcam Input**: Built-in webcam snapshot capture.
- 🧠 **Adaptive Illumination & CLAHE**: Uses Contrast Limited Adaptive Histogram Equalization to ensure fair, equitable, and reliable face detection across all skin tones and challenging lighting conditions.
- 🎯 **Multi-Scale Detection**: Merges multiple cascade classifiers for high sensitivity on both single-person close-ups and crowd/group photos.
- 🖼️ **Detected Faces Inspector**: Automatically crops and presents each detected face in an isolated gallery card with dimension tags.
- 💾 **Annotated Image Export**: 1-click download of the annotated image with custom accent colors.
- ☁️ **Deployable for Free**: 1-click free deployment on **Streamlit Community Cloud**.

---

## 💻 Local Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
streamlit run app.py
```
Your browser will open automatically at: `http://localhost:8501`

---

## 🚀 Free 1-Click Deployment to Streamlit Cloud

To host your Python application online for free:

1. Push your code to your GitHub repository:
   ```bash
   git add .
   git commit -m "Pure Python Face Detection App"
   git push origin main
   ```
2. Go to **[share.streamlit.io](https://share.streamlit.io)** (log in with your GitHub account).
3. Click **"New app"**.
4. Select your repository, set the branch to `main`, and main file path to `app.py`.
5. Click **"Deploy!"**.
6. Within 1–2 minutes, your live Python application will be online with a permanent URL:
   ```
   https://<your-app-name>.streamlit.app
   ```

---

## 🏗️ Project Architecture

```
ML-Project2/
│
├── app.py                  # Main Streamlit Python web application
├── requirements.txt        # Python package dependencies
├── .gitignore              # Git ignore rules
└── README.md               # Project documentation
```

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
