# 👤 FaceVision AI - Pure Python Face Detection Application

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/CV-OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **FaceVision AI** is a pure Python-powered computer vision web application for detecting human faces in real-time. Built with **Streamlit** and **OpenCV**, it features state-of-the-art **AI Deep Learning (YuNet CNN)**, 5-point facial landmark tracking, Non-Maximum Suppression with containment filtering, webcam capture, local image upload, and an isolated face crops inspector gallery.

---

## 🌟 Key Features

- 🤖 **AI Deep Learning Engine (YuNet CNN)**: Robust neural network face detector trained on WIDER FACE. Handles tilted heads, smiles, occlusions, and diverse lighting with zero neck/collar false positives.
- 🎯 **5-Point Facial Landmarks**: Tracks eyes, nose tip, and mouth corners in real-time.
- 🔍 **Classic Haar Cascade Fallback**: Includes tuned Viola-Jones multi-pass cascades with containment Non-Maximum Suppression.
- 📁 **Local Image Upload**: Directly upload images (PNG, JPG, WebP) from your computer.
- 📷 **Live Webcam Input**: Capture snapshots straight from your webcam.
- 👥 **Detected Faces Inspector**: Automatically crops and presents each detected face in an isolated gallery card with dimension tags and confidence ratings.
- 📥 **Annotated Image Export**: 1-click download of the annotated image with custom accent colors.
- 🚀 **Deployable for Free**: 1-click free deployment on **Streamlit Community Cloud**.

---

## 🚀 Local Quickstart

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

## ☁️ Free 1-Click Deployment to Streamlit Cloud

To host your Python application online for free:

1. Push your code to your GitHub repository:
   ```bash
   git add .
   git commit -m "Update Face Detection App with YuNet AI"
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

## 📂 Project Architecture

```
face-detection-app/
├── app.py                            # Streamlit web application (YuNet AI + Haar Cascade)
├── face_detection_yunet_2023mar.onnx  # Pretrained lightweight YuNet ONNX model (232 KB)
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
└── README.md                         # Documentation & Quickstart
```

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
