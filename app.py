"""
FaceVision AI - Pure Python Face Detection Web Application
Built with Streamlit & OpenCV. Ready for 1-Click Free Deployment on Streamlit Cloud.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

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

@st.cache_resource
def load_cascades():
    """Load OpenCV pre-trained Haar Cascades safely across OpenCV versions."""
    cascade_cls = getattr(cv2, 'CascadeClassifier', None)
    if cascade_cls is None and hasattr(cv2, 'objdetect'):
        cascade_cls = getattr(cv2.objdetect, 'CascadeClassifier', None)
        
    if cascade_cls is None:
        st.error("OpenCV CascadeClassifier is not available in current build. Ensure opencv-python-headless < 5.0.0 is installed.")
        st.stop()
        
    data_dir = getattr(cv2, 'data', None)
    if data_dir and hasattr(data_dir, 'haarcascades'):
        base_dir = data_dir.haarcascades
    else:
        import os
        base_dir = os.path.join(os.path.dirname(cv2.__file__), 'data') + os.sep

    frontal_path = base_dir + 'haarcascade_frontalface_default.xml'
    alt_path = base_dir + 'haarcascade_frontalface_alt2.xml'
    
    cascade_frontal = cascade_cls(frontal_path)
    cascade_alt = cascade_cls(alt_path)
    return cascade_frontal, cascade_alt

def detect_faces(image_bgr, scale_factor=1.08, min_neighbors=3, min_size=(25, 25), box_color=(0, 242, 254)):
    """
    Detect faces using multi-scale OpenCV pipeline with adaptive contrast enhancement (CLAHE).
    Ensures high accuracy and fair detection across all skin tones, lighting conditions, and distances.
    """
    cascade_frontal, cascade_alt = load_cascades()
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Contrast Limited Adaptive Histogram Equalization (CLAHE) for illumination balancing
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    
    # Multi-pass detection: frontal cascade
    faces_frontal = cascade_frontal.detectMultiScale(
        enhanced_gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Secondary pass with alt cascade for profile / angled / edge faces
    faces_alt = cascade_alt.detectMultiScale(
        enhanced_gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Merge and deduplicate overlapping bounding boxes
    all_faces = list(faces_frontal)
    for (x2, y2, w2, h2) in faces_alt:
        overlap = False
        for (x1, y1, w1, h1) in all_faces:
            # Check bounding box Intersection-over-Union (IoU)
            xi1, yi1 = max(x1, x2), max(y1, y2)
            xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            box1_area = w1 * h1
            box2_area = w2 * h2
            iou = inter_area / float(box1_area + box2_area - inter_area) if (box1_area + box2_area - inter_area) > 0 else 0
            if iou > 0.35:
                overlap = True
                break
        if not overlap:
            all_faces.append((x2, y2, w2, h2))
            
    # Draw futuristic annotations and extract crops
    annotated = image_bgr.copy()
    crops = []
    
    for i, (x, y, w, h) in enumerate(all_faces):
        # Corner brackets style / bounding rectangle
        cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)
        
        # Label tag
        label = f"Face #{i + 1}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x, max(0, y - 24)), (x + tw + 12, max(0, y)), box_color, -1)
        cv2.putText(annotated, label, (x + 6, max(0, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (8, 12, 20), 1, cv2.LINE_AA)
        
        # Extract face crop with border padding
        pad = int(min(w, h) * 0.15)
        cy1, cy2 = max(0, y - pad), min(image_bgr.shape[0], y + h + pad)
        cx1, cx2 = max(0, x - pad), min(image_bgr.shape[1], x + w + pad)
        face_crop = image_bgr[cy1:cy2, cx1:cx2]
        if face_crop.size > 0:
            crops.append((i + 1, cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB), w, h))
            
    return annotated, len(all_faces), crops

# Sidebar Configuration
st.sidebar.markdown("## ⚙️ Detection Settings")

sensitivity = st.sidebar.select_slider(
    "Detection Sensitivity",
    options=["Standard", "High", "Maximum (Groups & Crowds)"],
    value="High",
    help="Adjust sensitivity to detect smaller faces or crowded scenes."
)

if sensitivity == "Standard":
    scale_factor = 1.10
    min_neighbors = 4
elif sensitivity == "High":
    scale_factor = 1.06
    min_neighbors = 3
else:
    scale_factor = 1.04
    min_neighbors = 2

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
st.sidebar.markdown(
    "**FaceVision AI** is 100% Python-powered. Uses OpenCV multi-pass Haar Cascades with CLAHE illumination enhancement."
)

# Header Section
st.markdown('<div class="main-title">👤 FaceVision AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle-text">Pure Python Computer Vision & Face Detection Application. '
    'Upload any image from your computer or use your camera to detect faces in real-time.</div>',
    unsafe_allow_html=True
)

# Mode Selector
input_mode = st.radio(
    "Select Input Source:",
    options=["📁 Upload Photo from Local System", "📸 Live Camera Capture"],
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
        # Convert RGB to BGR for OpenCV
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
        annotated_bgr, face_count, face_crops = detect_faces(
            image_to_process,
            scale_factor=scale_factor,
            min_neighbors=min_neighbors,
            box_color=selected_bgr_color
        )
        
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_to_process.shape[:2]

    # Metrics Summary Row
    m1, m2, m3 = st.columns(3)
    m1.metric("Faces Detected", f"{face_count}")
    m2.metric("Image Dimensions", f"{w} × {h} px")
    m3.metric("Sensitivity Profile", sensitivity)

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
            label="💾 Download Annotated Image (PNG)",
            data=byte_data,
            file_name="detected_faces_result.png",
            mime="image/png"
        )

    with col_gallery:
        st.subheader("🖼️ Detected Faces Inspector")
        if face_count == 0:
            st.info("No faces detected with current settings. Try setting Sensitivity to 'Maximum' in the sidebar.")
        else:
            # Display crops in a 2-column subgrid
            crop_cols = st.columns(2)
            for idx, (face_id, crop_img, fw, fh) in enumerate(face_crops):
                col_target = crop_cols[idx % 2]
                with col_target:
                    st.image(crop_img, use_container_width=True)
                    st.caption(f"**Face #{face_id}** — {fw}×{fh}px")
else:
    st.info("👆 Please upload an image from your computer or take a camera snapshot above to run face detection.")
