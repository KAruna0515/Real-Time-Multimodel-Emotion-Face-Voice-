import streamlit as st
import requests
import cv2
import os
import time
import platform

API_URL = os.getenv("MM_API_URL", "http://127.0.0.1:8000/analyze")

def analyze_frame(frame, device_selection):
    """Encodes frame and sends it to the backend API for analysis."""
    _, buf = cv2.imencode('.jpg', frame)
    files = {
        "audio": ("chunk.wav", b"", "audio/wav"),
        "frame": ("frame.jpg", buf.tobytes(), "image/jpeg")
    }
    payload = {"device": device_selection}
    try:
        response = requests.post(API_URL, files=files, data=payload, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None

def get_camera():
    """Initializes and returns a camera object, cached in session state."""
    if 'camera' not in st.session_state:
        # Use CAP_DSHOW on Windows for better performance, otherwise default
        cap_flag = cv2.CAP_DSHOW if platform.system() == "Windows" else 0
        st.session_state.camera = cv2.VideoCapture(0, cap_flag)
    return st.session_state.camera

def release_camera():
    """Releases the camera resource if it exists in session state."""
    if 'camera' in st.session_state:
        st.session_state.camera.release()
        del st.session_state.camera
st.title("Realtime Multimodal Emotion Demo")

st.sidebar.header("Settings")
interval = st.sidebar.slider("Send interval (s)", 1.0, 5.0, 2.0, 0.5)
device = st.sidebar.selectbox("Device", ["cpu", "cuda"])

run = st.sidebar.checkbox("Start Realtime Analysis")

frame_placeholder = st.empty()
results_placeholder = st.empty()

if run:
    cap = get_camera()
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from camera. Please ensure it's not in use by another application.")
                release_camera()
                break

            # Display the current frame
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            # Analyze the frame and display results
            analysis_result = analyze_frame(frame, device)
            results_placeholder.json(analysis_result or {"status": "Waiting for next analysis..."})

            time.sleep(interval)
    finally:
        release_camera()
else:
    release_camera()
    frame_placeholder.empty()
    results_placeholder.empty()
