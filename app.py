import cv2
import pandas as pd
import streamlit as st
from datetime import datetime
from PIL import Image
from ultralytics import YOLO
import os
import time

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Automated Safety Compliance Monitor",
    page_icon="🛡️",
    layout="wide"
)

# ---------------------------------------------------------
# Constants & Model Loading
# ---------------------------------------------------------
MODEL_PATH = "best.pt"
LOG_FILE = "violations_log.csv"

# Initialize CSV log file if it doesn't exist
if not os.path.exists(LOG_FILE):
    df = pd.DataFrame(columns=["Timestamp", "Source", "Missing_PPE", "Detected_Classes"])
    df.to_csv(LOG_FILE, index=False)

@st.cache_resource
def load_yolo_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found in the project directory. Place 'best.pt' here.")
        st.stop()
    return YOLO(model_path)

model = load_yolo_model(MODEL_PATH)

# Target Classes from dataset: 0: Helmet, 1: Mask, 2: Safety Vest, 3: boots, 4: glove
CLASS_NAMES = model.names

# ---------------------------------------------------------
# UI Header & Sidebar Controls
# ---------------------------------------------------------
st.title("🛡️ Automated Safety Compliance Monitor")
st.caption("Real-Time Computer Vision Pipeline for Physical Security & Workplace Safety")

st.sidebar.header("⚙️ Configuration")
mode = st.sidebar.selectbox("Select Mode", ["Real-Time CCTV Stream", "Upload Image", "Upload Video", "Daily Compliance Reports"])

# Set optimal confidence range (0.10 to 0.90, default 0.35)
conf_threshold = st.sidebar.slider("Detection Confidence", 0.10, 0.90, 0.35, 0.05)

# Mandatory Safety Rules Config
st.sidebar.subheader("🚨 Mandatory Safety Requirements")
require_helmet = st.sidebar.checkbox("Require Helmet", value=True)
require_vest = st.sidebar.checkbox("Require Safety Vest", value=True)
require_mask = st.sidebar.checkbox("Require Mask", value=False)

# Tracking timestamp to avoid logging every millisecond during continuous streams
if "last_log_time" not in st.session_state:
    st.session_state.last_log_time = 0

# Helper function to check compliance and log violations
def evaluate_compliance(source_label, detected_class_names, is_stream=False):
    # Case 1: Empty Scene (No detections made at all)
    if not detected_class_names:
        return [], "No Equipment Detected"

    missing_ppe = []
    if require_helmet and 'Helmet' not in detected_class_names:
        missing_ppe.append("Helmet")
    if require_vest and 'Safety Vest' not in detected_class_names:
        missing_ppe.append("Safety Vest")
    if require_mask and 'Mask' not in detected_class_names:
        missing_ppe.append("Mask")
        
    # Case 2: Safety Violation Found
    if missing_ppe:
        current_time = time.time()
        # Cooldown logic: Log at most once every 3 seconds for continuous video/CCTV streams
        if not is_stream or (current_time - st.session_state.last_log_time > 3.0):
            missing_str = ", ".join(missing_ppe)
            detected_str = ", ".join(detected_class_names)
            new_entry = pd.DataFrame([{
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Source": source_label,
                "Missing_PPE": missing_str,
                "Detected_Classes": detected_str
            }])
            new_entry.to_csv(LOG_FILE, mode='a', header=False, index=False)
            st.session_state.last_log_time = current_time
            
        return missing_ppe, "Violation"
        
    # Case 3: Fully Compliant
    return [], "Compliant"

# ---------------------------------------------------------
# Mode 1: Real-Time CCTV Stream
# ---------------------------------------------------------
if mode == "Real-Time CCTV Stream":
    st.subheader("📹 Real-Time Stream Monitoring")
    stream_source = st.text_input("Enter RTSP Stream URL or Camera ID (0 for Webcam):", "0")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        start_btn = st.button("▶️ Start Stream", use_container_width=True)
        stop_btn = st.button("⏹️ Stop Stream", use_container_width=True)
        alert_placeholder = st.empty()
        metrics_placeholder = st.empty()

    if start_btn:
        source = int(stream_source) if stream_source.isdigit() else stream_source
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            st.error(f"Failed to open video source: {stream_source}")
        else:
            with col1:
                frame_window = st.image([])
            
            while cap.isOpened() and not stop_btn:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Video stream interrupted or ended.")
                    break
                
                # Model Inference
                results = model(frame, conf=conf_threshold, verbose=False)
                annotated_frame = results[0].plot()
                
                # Extract detected classes
                detected_ids = results[0].boxes.cls.cpu().numpy().astype(int) if len(results[0].boxes) > 0 else []
                detected_names = list(set([CLASS_NAMES[i] for i in detected_ids]))
                
                # Check for violations
                violations, status = evaluate_compliance("CCTV Stream", detected_names, is_stream=True)
                
                # Display status & alerts
                if status == "Violation":
                    alert_placeholder.error(f"🚨 **SAFETY VIOLATION!**\nMissing: {', '.join(violations)}")
                elif status == "Compliant":
                    alert_placeholder.success("✅ **Compliance Status:** All mandatory PPE present.")
                else:
                    alert_placeholder.info("ℹ️ **Status:** No workers/PPE detected in scene.")
                
                metrics_placeholder.markdown("**Detected Equipment:**\n" + ("\n".join([f"- {item}" for item in detected_names]) if detected_names else "- None"))
                
                # Render Frame
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_window.image(rgb_frame, channels="RGB", use_container_width=True)
            
            cap.release()

# ---------------------------------------------------------
# Mode 2: Upload Image
# ---------------------------------------------------------
elif mode == "Upload Image":
    st.subheader("🖼️ Image Compliance Inspector")
    uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
        # Inference
        results = model(image, conf=conf_threshold)
        annotated_image = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        detected_ids = results[0].boxes.cls.cpu().numpy().astype(int) if len(results[0].boxes) > 0 else []
        detected_names = list(set([CLASS_NAMES[i] for i in detected_ids]))
        
        violations, status = evaluate_compliance(uploaded_file.name, detected_names, is_stream=False)
        
        with col2:
            st.image(annotated_rgb, caption="Detection Analysis", use_container_width=True)
            st.markdown("### Compliance Summary")
            if status == "Violation":
                st.error(f"❌ **Non-Compliant! Missing:** {', '.join(violations)}")
            elif status == "Compliant":
                st.success("✅ **Fully Compliant!**")
            else:
                st.info("ℹ️ **No PPE items detected in image.**")
                
            st.write(f"**Detected Equipment:** {', '.join(detected_names) if detected_names else 'None'}")

# ---------------------------------------------------------
# Mode 3: Upload Video
# ---------------------------------------------------------
elif mode == "Upload Video":
    st.subheader("🎬 Video File Compliance Inspector")
    uploaded_video = st.file_uploader("Upload a video snippet...", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        temp_video_path = f"temp_{uploaded_video.name}"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.read())
            
        cap = cv2.VideoCapture(temp_video_path)
        st_frame = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            results = model(frame, conf=conf_threshold, verbose=False)
            annotated_frame = results[0].plot()
            
            detected_ids = results[0].boxes.cls.cpu().numpy().astype(int) if len(results[0].boxes) > 0 else []
            detected_names = list(set([CLASS_NAMES[i] for i in detected_ids]))
            
            evaluate_compliance(uploaded_video.name, detected_names, is_stream=True)
            
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st_frame.image(rgb_frame, channels="RGB", use_container_width=True)
            
        cap.release()
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        st.success("Video processing complete!")

# ---------------------------------------------------------
# Mode 4: Daily Compliance Reports
# ---------------------------------------------------------
elif mode == "Daily Compliance Reports":
    st.subheader("📊 Violation Logs & Incident Analytics")
    
    if os.path.exists(LOG_FILE):
        log_df = pd.read_csv(LOG_FILE)
        
        if not log_df.empty:
            st.dataframe(log_df.sort_values(by="Timestamp", ascending=False), use_container_width=True)
            
            csv_data = log_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Daily Log (CSV)",
                data=csv_data,
                file_name=f"PPE_Compliance_Report_{datetime.now().strftime('%Y_%m_%d')}.csv",
                mime="text/csv"
            )
            
            st.markdown("---")
            st.subheader("Summary Metrics")
            m1, m2 = st.columns(2)
            m1.metric("Total Incidents Logged", len(log_df))
            
            mode_series = log_df["Missing_PPE"].mode()
            top_missing = mode_series[0] if not mode_series.empty else "None"
            m2.metric("Most Frequently Missing Equipment", top_missing)
        else:
            st.info("No compliance violations have been logged yet.")
    else:
        st.info("Log file initialized. Start monitoring to collect data.")