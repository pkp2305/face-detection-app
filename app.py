"""
FaceVision AI - Pure Python Face Detection Web Application
Built with Streamlit & OpenCV (Deep Learning YuNet + Haar Cascade fallback).
Ready for 1-Click Free Deployment on Streamlit Cloud.
"""

import os
import io
import urllib.request
import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="FaceVision AI - Face Detection",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Cyberpunk & Glassmorphic CSS Theme
st.markdown("""
<style>
    /* Dark Cyber Theme Styling */
    .stApp {
        background-color: #080c14;
        color: #f0f4fc;
    }
    
    /* Header Gradient */
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 50%, #7f00ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .subtitle-text {
        color: #8fa0bd;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    
    /* Metric Card Styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #00f2fe;
        font-weight: 700;
    }
    
    /* Custom Card */
    .info-card {
        background: rgba(18, 26, 43, 0.7);
        border: 1px solid rgba(0, 242, 254, 0.25);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    
    /* Button overrides */
    .stButton>button {
        background: linear-gradient(135deg, #00f2fe, #4facfe);
        color: #050b14;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        filter: brightness(1.15);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

MODEL_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"

@st.cache_resource
def get_yunet_model_path():
    """Ensure YuNet ONNX model is present locally, downloading if necessary."""
    local_path = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)
    if not os.path.exists(local_path) or os.path.getsize(local_path) < 100000:
        try:
            req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(local_path, "wb") as f:
                f.write(resp.read())
        except Exception as e:
            st.warning(f"Could not download YuNet model automatically: {e}. Falling back to Haar cascades.")
            return None
    return local_path

@st.cache_resource
def load_yunet_detector(model_path, conf_threshold=0.55, nms_threshold=0.3):
    """Load OpenCV built-in Deep Learning FaceDetectorYN."""
    if not model_path or not os.path.exists(model_path):
        return None
    try:
        detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(320, 320),
            score_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            top_k=5000
        )
        return detector
    except Exception as e:
        st.error(f"Failed to initialize YuNet detector: {e}")
        return None

@st.cache_resource
def load_cascades():
    """Load OpenCV pre-trained Haar Cascades as fallback."""
    cascade_cls = getattr(cv2, 'CascadeClassifier', None)
    if cascade_cls is None and hasattr(cv2, 'objdetect'):
        cascade_cls = getattr(cv2.objdetect, 'CascadeClassifier', None)
        
    if cascade_cls is None:
        return None, None
        
    data_dir = getattr(cv2, 'data', None)
    if data_dir and hasattr(data_dir, 'haarcascades'):
        base_dir = data_dir.haarcascades
    else:
        base_dir = os.path.join(os.path.dirname(cv2.__file__), 'data') + os.sep

    frontal_path = base_dir + 'haarcascade_frontalface_default.xml'
    alt_path = base_dir + 'haarcascade_frontalface_alt2.xml'
    
    cascade_frontal = cascade_cls(frontal_path)
    cascade_alt = cascade_cls(alt_path)
    return cascade_frontal, cascade_alt

def non_max_suppression_boxes(boxes, iou_thresh=0.28, containment_thresh=0.38):
    """Deduplicate overlapping and nested bounding boxes for Haar cascade."""
    if len(boxes) == 0:
        return []
    sorted_boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept_boxes = []
    
    for box_a in sorted_boxes:
        xa1, ya1, wa, ha = box_a
        xa2, ya2 = xa1 + wa, ya1 + ha
        area_a = wa * ha
        is_duplicate = False
        
        for box_b in kept_boxes:
            xb1, yb1, wb, hb = box_b
            xb2, yb2 = xb1 + wb, yb1 + hb
            area_b = wb * hb
            
            xi1, yi1 = max(xa1, xb1), max(ya1, yb1)
            xi2, yi2 = min(xa2, xb2), min(ya2, yb2)
            inter_w = max(0, xi2 - xi1)
            inter_h = max(0, yi2 - yi1)
            inter_area = inter_w * inter_h
            
            if inter_area > 0:
                union_area = area_a + area_b - inter_area
                iou = inter_area / float(union_area) if union_area > 0 else 0
                min_area = min(area_a, area_b)
                containment = inter_area / float(min_area) if min_area > 0 else 0
                
                if iou > iou_thresh or containment > containment_thresh:
                    is_duplicate = True
                    break
                    
        if not is_duplicate:
            kept_boxes.append(box_a)
            
    return sorted(kept_boxes, key=lambda b: b[0])

def detect_faces_yunet(image_bgr, model_path, conf_threshold=0.55, show_landmarks=True, box_color=(0, 242, 254)):
    """
    Detect faces using OpenCV YuNet Deep Learning CNN.
    Trained on WIDER FACE: extremely robust to tilted heads, smiling, diverse skin tones,
    and lighting. Eliminates false positives on necks, clothing, and background textures.
    """
    h_img, w_img = image_bgr.shape[:2]
    detector = cv2.FaceDetectorYN.create(
        model=model_path,
        config="",
        input_size=(w_img, h_img),
        score_threshold=conf_threshold,
        nms_threshold=0.3,
        top_k=5000
    )
    detector.setInputSize((w_img, h_img))
    _, raw_faces = detector.detect(image_bgr)
    
    annotated = image_bgr.copy()
    crops = []
    detected_count = 0
    
    if raw_faces is not None and len(raw_faces) > 0:
        # Sort faces from left to right for intuitive numbering
        raw_faces = sorted(raw_faces, key=lambda f: f[0])
        detected_count = len(raw_faces)
        
        for i, face in enumerate(raw_faces):
            x = max(0, int(face[0]))
            y = max(0, int(face[1]))
            w = max(1, int(face[2]))
            h = max(1, int(face[3]))
            score = float(face[14])
            
            # Boundary clamp
            x_end = min(w_img, x + w)
            y_end = min(h_img, y + h)
            w = x_end - x
            h = y_end - y
            
            # Bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)
            
            # Label with confidence
            label = f"Face #{i + 1} ({int(score * 100)}%)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x, max(0, y - 24)), (x + tw + 12, max(0, y)), box_color, -1)
            cv2.putText(annotated, label, (x + 6, max(0, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (8, 12, 20), 1, cv2.LINE_AA)
            
            # Optional 5 Facial Landmarks (Right eye, Left eye, Nose tip, Right mouth, Left mouth)
            if show_landmarks and len(face) >= 14:
                landmark_colors = [
                    (0, 0, 255),    # Right eye - Red
                    (0, 255, 0),    # Left eye - Green
                    (255, 255, 0),  # Nose - Cyan
                    (255, 0, 255),  # Right mouth - Magenta
                    (0, 255, 255)   # Left mouth - Yellow
                ]
                for p_idx in range(5):
                    px = int(face[4 + p_idx * 2])
                    py = int(face[5 + p_idx * 2])
                    if 0 <= px < w_img and 0 <= py < h_img:
                        cv2.circle(annotated, (px, py), 3, landmark_colors[p_idx], -1)
            
            # Extract face crop with border padding
            pad = int(min(w, h) * 0.15)
            cy1, cy2 = max(0, y - pad), min(h_img, y + h + pad)
            cx1, cx2 = max(0, x - pad), min(w_img, x + w + pad)
            face_crop = image_bgr[cy1:cy2, cx1:cx2]
            if face_crop.size > 0:
                crops.append((i + 1, cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB), w, h, score))
                
    return annotated, detected_count, crops

def detect_faces_haar(image_bgr, scale_factor=1.12, min_neighbors=6, min_size=(50, 50), box_color=(0, 242, 254), enable_dual=False):
    """Fallback detector using classic Haar cascades with containment suppression."""
    cascade_frontal, cascade_alt = load_cascades()
    if cascade_frontal is None:
        return image_bgr, 0, []
        
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    denoised_gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    raw_frontal = cascade_frontal.detectMultiScale(
        denoised_gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    all_candidates = [tuple(b) for b in raw_frontal] if len(raw_frontal) > 0 else []
    
    if enable_dual and cascade_alt is not None:
        raw_alt = cascade_alt.detectMultiScale(
            denoised_gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(raw_alt) > 0:
            all_candidates.extend([tuple(b) for b in raw_alt])
            
    clean_faces = non_max_suppression_boxes(all_candidates, iou_thresh=0.28, containment_thresh=0.38)
    annotated = image_bgr.copy()
    crops = []
    
    for i, (x, y, w, h) in enumerate(clean_faces):
        cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)
        label = f"Face #{i + 1}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x, max(0, y - 24)), (x + tw + 12, max(0, y)), box_color, -1)
        cv2.putText(annotated, label, (x + 6, max(0, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (8, 12, 20), 1, cv2.LINE_AA)
        
        pad = int(min(w, h) * 0.15)
        cy1, cy2 = max(0, y - pad), min(image_bgr.shape[0], y + h + pad)
        cx1, cx2 = max(0, x - pad), min(image_bgr.shape[1], x + w + pad)
        face_crop = image_bgr[cy1:cy2, cx1:cx2]
        if face_crop.size > 0:
            crops.append((i + 1, cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB), w, h, None))
            
    return annotated, len(clean_faces), crops

# Sidebar Configuration
st.sidebar.markdown("## ⚙️ Detection Settings")

engine_choice = st.sidebar.radio(
    "Detection Engine",
    options=["🤖 AI Deep Learning (YuNet CNN - Recommended)", "🔍 Classic Haar Cascade (Legacy)"],
    index=0,
    help="AI Deep Learning detects faces at any angle, tilted heads, smiles, and avoids neck/shirt false positives."
)

if "AI Deep Learning" in engine_choice:
    conf_thresh = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.30,
        max_value=0.95,
        value=0.55,
        step=0.05,
        help="Higher values reject lower confidence shapes. 0.50-0.60 is recommended for portraits & webcams."
    )
    show_landmarks = st.sidebar.checkbox(
        "Show Facial Landmarks",
        value=True,
        help="Displays 5 facial landmark keypoints (eyes, nose tip, mouth corners)."
    )
else:
    preset_haar = st.sidebar.selectbox(
        "Haar Preset",
        options=["Portrait / Single Face", "Group / Crowd Scene", "Custom"]
    )
    if preset_haar == "Portrait / Single Face":
        scale_factor = 1.15
        min_neighbors = 7
        min_size_px = 60
        enable_dual = False
    elif preset_haar == "Group / Crowd Scene":
        scale_factor = 1.10
        min_neighbors = 5
        min_size_px = 35
        enable_dual = True
    else:
        scale_factor = st.sidebar.slider("Scale Factor", 1.05, 1.30, 1.12, 0.01)
        min_neighbors = st.sidebar.slider("Min Neighbors", 3, 12, 6, 1)
        min_size_px = st.sidebar.slider("Min Size (px)", 20, 200, 50, 5)
        enable_dual = st.sidebar.checkbox("Enable Dual Cascade", value=False)

color_choice = st.sidebar.selectbox(
    "Overlay Accent Color",
    options=["Cyan (#00f2fe)", "Emerald (#00f5a0)", "Coral (#ff3366)", "Gold (#ffb703)"]
)

color_map = {
    "Cyan (#00f2fe)": (254, 242, 0),    # BGR format
    "Emerald (#00f5a0)": (160, 245, 0),
    "Coral (#ff3366)": (102, 51, 255),
    "Gold (#ffb703)": (3, 183, 255)
}
selected_bgr_color = color_map[color_choice]

st.sidebar.markdown("---")
if "AI Deep Learning" in engine_choice:
    st.sidebar.success("⚡ Powered by **OpenCV YuNet CNN**. Immune to collar/neck false positives.")
else:
    st.sidebar.info("ℹ️ Using **Haar Cascades**. Best when faces are strictly upright and frontal.")

# Header Section
st.markdown('<div class="main-title">👤 FaceVision AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle-text">Pure Python Computer Vision & Face Detection Application. '
    'High precision face detection supporting tilted angles, expressions, and zero neck false-positives.</div>',
    unsafe_allow_html=True
)

# Mode Selector
input_mode = st.radio(
    "Select Input Source:",
    options=["📁 Upload Photo from Local System", "📷 Live Camera Capture"],
    horizontal=True
)

image_to_process = None

if input_mode == "📁 Upload Photo from Local System":
    uploaded_file = st.file_uploader(
        "Choose an image from your computer (JPG, PNG, WebP):",
        type=["jpg", "jpeg", "png", "webp"],
        help="Select any image file from your local disk."
    )
    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")
        image_to_process = np.array(pil_image)
        image_to_process = cv2.cvtColor(image_to_process, cv2.COLOR_RGB2BGR)
else:
    camera_file = st.camera_input("Take a photo with your webcam:")
    if camera_file is not None:
        pil_image = Image.open(camera_file).convert("RGB")
        image_to_process = np.array(pil_image)
        image_to_process = cv2.cvtColor(image_to_process, cv2.COLOR_RGB2BGR)

# Process Image
if image_to_process is not None:
    with st.spinner("Analyzing image and detecting faces..."):
        if "AI Deep Learning" in engine_choice:
            model_path = get_yunet_model_path()
            if model_path:
                annotated_bgr, face_count, face_crops = detect_faces_yunet(
                    image_to_process,
                    model_path=model_path,
                    conf_threshold=conf_thresh,
                    show_landmarks=show_landmarks,
                    box_color=selected_bgr_color
                )
            else:
                st.warning("YuNet model unavailable. Using Haar Cascade fallback.")
                annotated_bgr, face_count, face_crops = detect_faces_haar(
                    image_to_process,
                    box_color=selected_bgr_color
                )
        else:
            annotated_bgr, face_count, face_crops = detect_faces_haar(
                image_to_process,
                scale_factor=scale_factor,
                min_neighbors=min_neighbors,
                min_size=(min_size_px, min_size_px),
                box_color=selected_bgr_color,
                enable_dual=enable_dual
            )
        
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_to_process.shape[:2]

    # Metrics Summary Row
    m1, m2, m3 = st.columns(3)
    m1.metric("Faces Detected", f"{face_count}")
    m2.metric("Image Dimensions", f"{w} × {h} px")
    engine_label = "YuNet CNN (AI)" if "AI Deep Learning" in engine_choice else "Haar Cascade"
    m3.metric("Detection Engine", engine_label)

    st.markdown("---")

    # Display Annotated Image
    col_main, col_gallery = st.columns([1.6, 1.0])

    with col_main:
        st.subheader("🎯 Annotated Detection Result")
        st.image(annotated_rgb, use_container_width=True)

        # Download annotated image button
        pil_annotated = Image.fromarray(annotated_rgb)
        buf = io.BytesIO()
        pil_annotated.save(buf, format="PNG")
        byte_data = buf.getvalue()

        st.download_button(
            label="📥 Download Annotated Image (PNG)",
            data=byte_data,
            file_name="detected_faces_result.png",
            mime="image/png"
        )

    with col_gallery:
        st.subheader("👥 Detected Faces Inspector")
        if face_count == 0:
            st.info("No faces detected. If using AI, try lowering the confidence threshold. If using Haar, ensure your head is upright.")
        else:
            # Display crops in a 2-column subgrid
            crop_cols = st.columns(2)
            for idx, (face_id, crop_img, fw, fh, score) in enumerate(face_crops):
                col_target = crop_cols[idx % 2]
                with col_target:
                    st.image(crop_img, use_container_width=True)
                    caption_text = f"**Face #{face_id}** — {fw}×{fh}px"
                    if score is not None:
                        caption_text += f" ({int(score * 100)}%)"
                    st.caption(caption_text)
else:
    st.info("👋 Please upload an image from your computer or take a camera snapshot above to run face detection.")
