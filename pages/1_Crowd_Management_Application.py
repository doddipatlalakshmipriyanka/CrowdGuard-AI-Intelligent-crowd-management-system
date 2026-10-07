import os
import time
import uuid
import threading
import tempfile
import smtplib
from pathlib import Path
from email.message import EmailMessage

import av
import cv2
import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from aiortc.contrib.media import MediaRecorder
from streamlit_folium import st_folium
from streamlit_js_eval import streamlit_js_eval


# ============================================================
# PATHS / PAGE CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = BASE_DIR / "image"
RECORD_DIR = BASE_DIR / "camera_records"
RECORD_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="CrowdGuard AI - Intelligent Crowd Management System",
    page_icon="👥",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 1100px; padding-top: 1.5rem; }
    .metric-card {
        background: rgba(7, 25, 55, .78);
        border: 1px solid rgba(90, 180, 255, .30);
        border-radius: 14px;
        padding: 12px;
    }
    /* Professional white sidebar */
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    section[data-testid="stSidebar"] > div {
        background: #ffffff;
    }

    /* Sidebar typography */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #172033;
    }

    /* Selectbox and number-input cards */
    section[data-testid="stSidebar"] [data-testid="stSelectbox"],
    section[data-testid="stSidebar"] [data-testid="stNumberInput"] {
        background: #f8fafc;
        border: 1px solid #dbe3ee;
        border-radius: 12px;
        padding: 8px 10px 10px;
        margin-bottom: 10px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
    }

    section[data-testid="stSidebar"] [data-testid="stSelectbox"]:hover,
    section[data-testid="stSidebar"] [data-testid="stNumberInput"]:hover {
        border-color: #93c5fd;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.10);
    }

    section[data-testid="stSidebar"] input {
        background: #ffffff !important;
        color: #172033 !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
    }

    section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #ffffff;
        border-color: #cbd5e1;
        border-radius: 8px;
        color: #172033;
    }

    /* Adaptive threshold metric card */
    section[data-testid="stSidebar"] [data-testid="stMetric"] {
        background: linear-gradient(135deg, #eff6ff, #ffffff);
        border: 1px solid #bfdbfe;
        border-radius: 14px;
        padding: 12px;
        margin: 12px 0;
        box-shadow: 0 4px 12px rgba(30, 64, 175, 0.08);
    }

    section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #475569 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #1d4ed8 !important;
        font-weight: 700;
    }

    /* Sidebar info cards */
    section[data-testid="stSidebar"] [data-testid="stAlert"] {
        border-radius: 12px;
        border: 1px solid #dbeafe;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }

    /* Divider */
    section[data-testid="stSidebar"] hr {
        border-color: #e2e8f0;
    }

    /* Sidebar title */
    section[data-testid="stSidebar"] h1 {
        color: #0f172a !important;
        font-weight: 800;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

# These are project monitoring profiles, not official legal safety limits.
# The threshold is calculated from the selected place profile and monitored area.
PLACE_PROFILES = {
    "Classroom": {"density_limit": 1.50, "default_area": 50},
    "Bedroom": {"density_limit": 0.80, "default_area": 20},
    "Laboratory": {"density_limit": 1.20, "default_area": 60},
    "Corridor": {"density_limit": 1.80, "default_area": 30},
    "College Entrance": {"density_limit": 1.80, "default_area": 50},
    "College Hall": {"density_limit": 1.50, "default_area": 100},
    "Auditorium": {"density_limit": 1.20, "default_area": 200},
    "Canteen": {"density_limit": 1.30, "default_area": 80},
    "Playground / Open Area": {"density_limit": 0.80, "default_area": 250},
    "Bus / Transport Area": {"density_limit": 1.00, "default_area": 80},
    "General Indoor Area": {"density_limit": 1.40, "default_area": 50},
    "General Outdoor Area": {"density_limit": 0.80, "default_area": 120},
}

PLACE_KEYWORDS = {
    "Classroom": ["classroom", "class room", "class"],
    "Bedroom": ["bedroom", "bed room", "sleeping room"],
    "Laboratory": ["laboratory", "lab"],
    "Corridor": ["corridor", "passage", "walkway"],
    "College Entrance": ["entrance", "main gate", "gate"],
    "College Hall": ["college hall", "function hall", "hall"],
    "Auditorium": ["auditorium"],
    "Canteen": ["canteen", "cafeteria", "food court"],
    "Playground / Open Area": ["playground", "ground", "stadium", "park", "open area"],
    "Bus / Transport Area": ["bus", "bus stop", "transport", "station"],
}

CAMERA_CONFIDENCE = 0.40
CAMERA_IMAGE_SIZE = 640
HIGH_FRAMES_REQUIRED = 10
ALERT_COOLDOWN_SECONDS = 300
CAMERA_DISPLAY_WIDTH = 700


# ============================================================
# SESSION STATE
# ============================================================

def init_state():
    defaults = {
        "frame_store": {"frame": None, "lock": threading.Lock()},
        "camera_store": {
            "people": 0,
            "risk": "LOW",
            "high_frames": 0,
            "last_alert": 0.0,
            "lock": threading.Lock(),
        },
        "gps_store": {
            "latitude": None,
            "longitude": None,
            "accuracy": None,
            "location_text": None,
        },
        "threshold_store": {
            "threshold": 20,
            "place": "General Indoor Area",
            "area_m2": 50.0,
            "density_limit": PLACE_PROFILES["General Indoor Area"]["density_limit"],
            "lock": threading.Lock(),
        },
        "record_id": str(uuid.uuid4()),
        "captured_photo": None,
        "captured_photo_people": 0,
        "captured_photo_risk": "LOW",
        "captured_photo_threshold": 20,
        "live_photo_alert_sent": False,
        "uploaded_image_alert_id": None,
        "uploaded_video_alert_id": None,
        "gps_inferred_place": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()
frame_store = st.session_state.frame_store
camera_store = st.session_state.camera_store
gps_store = st.session_state.gps_store
threshold_store = st.session_state.threshold_store
record_file = RECORD_DIR / f"{st.session_state.record_id}_crowd_video.flv"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("👥 CrowdGuard AI")
    st.caption("AICW Capstone Project")
    st.markdown("---")
    st.markdown("### ⚙️ Adaptive Crowd Settings")
    st.caption("Thresholds are project monitoring values calculated from the selected place profile and monitored area.")

    place_options = ["Auto from GPS", *PLACE_PROFILES.keys()]

    def sync_place_profile():
        """Load the selected place profile into the editable sidebar fields."""
        selected = st.session_state.get("place_profile_select", "Auto from GPS")
        if selected == "Auto from GPS":
            selected = st.session_state.get("gps_inferred_place") or "General Indoor Area"
        profile = PLACE_PROFILES.get(selected, PLACE_PROFILES["General Indoor Area"])
        st.session_state["monitoring_area"] = float(profile["default_area"])
        st.session_state["density_limit"] = float(profile["density_limit"])

    if "monitoring_area" not in st.session_state:
        st.session_state["monitoring_area"] = float(PLACE_PROFILES["General Indoor Area"]["default_area"])
    if "density_limit" not in st.session_state:
        st.session_state["density_limit"] = float(PLACE_PROFILES["General Indoor Area"]["density_limit"])

    selected_place = st.selectbox(
        "📍 Monitoring Place",
        place_options,
        index=0,
        key="place_profile_select",
        on_change=sync_place_profile,
    )

    if selected_place == "Auto from GPS" and st.session_state.gps_inferred_place:
        effective_place = st.session_state.gps_inferred_place
        st.info(f"GPS suggestion: **{effective_place}**")
    elif selected_place == "Auto from GPS":
        effective_place = "General Indoor Area"
        st.info("No matching place was found from GPS text yet. Manual selection is recommended.")
    else:
        effective_place = selected_place

    profile = PLACE_PROFILES[effective_place]
    area_m2 = st.number_input(
        "📐 Monitored Area (m²)",
        min_value=5.0,
        max_value=5000.0,
        step=5.0,
        key="monitoring_area",
    )

    density_limit = st.number_input(
        "👥 Project Density Limit (people/m²)",
        min_value=0.10,
        max_value=10.0,
        step=0.10,
        key="density_limit",
        help="Project-defined monitoring setting. It is not presented as an official safety standard.",
    )

    adaptive_threshold = max(5, int(round(area_m2 * density_limit)))
    adaptive_threshold = min(adaptive_threshold, 5000)

    st.metric("🚨 Adaptive HIGH Threshold", f"{adaptive_threshold} people")
    st.caption(f"Formula: {area_m2:.0f} m² × {density_limit:.2f} people/m²")
    st.caption("Changing the monitoring place automatically loads that place's default area and density limit. You can still edit them manually afterward.")

    with threshold_store["lock"]:
        threshold_store["threshold"] = adaptive_threshold
        threshold_store["place"] = effective_place
        threshold_store["area_m2"] = float(area_m2)
        threshold_store["density_limit"] = float(density_limit)

    st.markdown("---")
    st.markdown("### 🧭 How adaptive monitoring works")
    st.write("1. Select the monitoring place.\n2. Set or review the monitored area.\n3. The system calculates a place-adaptive threshold.\n4. The camera counts people with YOLOv8.\n5. Risk is calculated from the adaptive threshold.")


# ============================================================
# CURRENT ADAPTIVE SETTINGS
# ============================================================

with threshold_store["lock"]:
    crowd_threshold = int(threshold_store["threshold"])
    current_place = threshold_store["place"]
    current_area = float(threshold_store["area_m2"])
    current_density = float(threshold_store["density_limit"])


# ============================================================
# TITLE / BANNER
# ============================================================

st.title("👥 CrowdGuard AI - Intelligent Crowd Management System")
st.write("AI-powered crowd detection, people counting, adaptive risk analysis, live photo/video capture and location-based HIGH-crowd alerts.")

st.markdown("---")
st.subheader("🏠 Welcome")

banner_candidates = [
    IMAGE_DIR / "banner (3).jpg",
    IMAGE_DIR / "banner (2).jpg",
    IMAGE_DIR / "banner.jpg",
]
banner_path = next((p for p in banner_candidates if p.exists()), None)
if banner_path:
    st.image(str(banner_path), use_container_width=True)
else:
    st.info("Banner image not found. Put a banner image inside the image folder if you want one displayed here.")

b1, b2, b3 = st.columns(3)
with b1:
    st.info("📷 **Live Photo**\n\nCapture the current processed camera frame.")
with b2:
    st.info("🎥 **Live Video**\n\nRecord the live camera stream.")
with b3:
    st.info("🚨 **HIGH Alert**\n\nSend crowd count and location by email.")


# ============================================================
# YOLO MODEL
# ============================================================

@st.cache_resource
def load_model():
    candidates = [BASE_DIR / "yolov8s.pt", BASE_DIR / "yolov8n.pt"]
    model_path = next((p for p in candidates if p.exists()), None)
    if model_path is None:
        raise FileNotFoundError(
            "YOLO model not found. Put yolov8s.pt (preferred) or yolov8n.pt in the project root."
        )
    return YOLO(str(model_path))


try:
    model = load_model()
except Exception as exc:
    st.error("❌ YOLO model could not be loaded.")
    st.code(str(exc))
    st.stop()


# ============================================================
# CORE HELPERS
# ============================================================

def get_risk(count: int, threshold: int) -> str:
    threshold = max(1, int(threshold))
    medium_threshold = max(1, threshold // 2)
    if count < medium_threshold:
        return "LOW"
    if count < threshold:
        return "MEDIUM"
    return "HIGH"


def risk_color(risk: str):
    if risk == "LOW":
        return (0, 255, 0)
    if risk == "MEDIUM":
        return (0, 255, 255)
    return (0, 0, 255)


def detect_people(image, conf=CAMERA_CONFIDENCE, imgsz=CAMERA_IMAGE_SIZE, iou=0.45, max_det=1000):
    results = model.predict(
        source=image,
        classes=[0],
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        max_det=max_det,
        verbose=False,
    )
    count = 0
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            try:
                class_id = int(box.cls[0])
            except Exception:
                continue
            if class_id != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])

            # Reject unusually narrow person detections. This helps prevent
            # objects/furniture from being counted as people in room scenes.
            box_width = x2 - x1
            box_height = y2 - y1

            if box_width <= 0 or box_height <= 0:
                continue

            aspect_ratio = box_width / box_height

            if aspect_ratio < 0.25:
                continue

            count += 1

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"Person {confidence:.2f}",
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
    return count


def infer_place_from_text(text: str):
    if not text:
        return None
    lowered = text.lower()
    for place, keywords in PLACE_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return place
    return None


def reverse_geocode(latitude, longitude, retries=3):
    """
    Convert GPS coordinates into a readable location name.

    Primary service: OpenStreetMap Nominatim.
    Fallback service: BigDataCloud reverse geocoding.

    The function tries to return a human-readable address/location text
    whenever possible. GPS coordinates remain valid even if both services
    are temporarily unavailable.
    """
    nominatim_url = "https://nominatim.openstreetmap.org/reverse"
    headers = {
        "User-Agent": "CrowdGuardAI/1.0 (crowd monitoring application)",
        "Accept": "application/json",
    }

    # -------------------------
    # 1. OpenStreetMap Nominatim
    # -------------------------
    zoom_levels = [18, 16, 14]
    last_error = None

    for attempt in range(retries):
        for zoom in zoom_levels:
            try:
                response = requests.get(
                    nominatim_url,
                    params={
                        "lat": latitude,
                        "lon": longitude,
                        "format": "jsonv2",
                        "zoom": zoom,
                        "addressdetails": 1,
                        "accept-language": "en",
                    },
                    headers=headers,
                    timeout=8,
                )

                if response.status_code == 200:
                    data = response.json() or {}

                    # Best result: complete readable address.
                    display_name = data.get("display_name")
                    if display_name:
                        return str(display_name)

                    # Fallback: construct readable text from address fields.
                    address = data.get("address", {}) or {}
                    parts = []
                    for key in [
                        "amenity",
                        "building",
                        "shop",
                        "neighbourhood",
                        "suburb",
                        "road",
                        "village",
                        "town",
                        "city",
                        "state_district",
                        "state",
                        "postcode",
                        "country",
                    ]:
                        value = address.get(key)
                        if value and value not in parts:
                            parts.append(str(value))

                    if parts:
                        return ", ".join(parts)

                    last_error = "Nominatim returned no readable address."
                    continue

                if response.status_code == 429:
                    last_error = "Nominatim rate limit (429)."
                    time.sleep(1.5 * (attempt + 1))
                    break

                if 500 <= response.status_code < 600:
                    last_error = f"Nominatim server error ({response.status_code})."
                    time.sleep(1.0 * (attempt + 1))
                    break

                last_error = f"Nominatim HTTP {response.status_code}."

            except (requests.RequestException, ValueError) as exc:
                last_error = str(exc)
                time.sleep(1.0 * (attempt + 1))

    # -------------------------
    # 2. BigDataCloud fallback
    # -------------------------
    fallback_url = "https://api.bigdatacloud.net/data/reverse-geocode-client"

    try:
        response = requests.get(
            fallback_url,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "localityLanguage": "en",
            },
            headers={"User-Agent": "CrowdGuardAI/1.0"},
            timeout=8,
        )

        if response.status_code == 200:
            data = response.json() or {}

            # Prefer the most specific useful locality first.
            parts = []
            for key in [
                "locality",
                "city",
                "district",
                "principalSubdivision",
                "countryName",
            ]:
                value = data.get(key)
                if value and value not in parts:
                    parts.append(str(value))

            # Some responses provide localityInfo with additional names.
            if not parts:
                locality_info = data.get("localityInfo", {}) or {}
                administrative = locality_info.get("administrative", []) or []
                for item in administrative:
                    name = item.get("name") if isinstance(item, dict) else None
                    if name and name not in parts:
                        parts.append(str(name))
                    if len(parts) >= 4:
                        break

            if parts:
                return ", ".join(parts)

            last_error = "Fallback geocoder returned no readable location."
        else:
            last_error = f"Fallback geocoder HTTP {response.status_code}."

    except (requests.RequestException, ValueError) as exc:
        last_error = str(exc)

    print("Reverse geocoding unavailable:", last_error)
    return None


# ============================================================
# EMAIL
# ============================================================

def load_email_config():
    try:
        secrets = st.secrets
        return {
            "sender_email": secrets.get("ALERT_EMAIL", ""),
            "sender_password": secrets.get("ALERT_PASSWORD", ""),
            "admin_email": secrets.get("ADMIN_EMAIL", ""),
        }
    except Exception:
        return {"sender_email": "", "sender_password": "", "admin_email": ""}


EMAIL_CONFIG = load_email_config()


def send_crowd_alert(people_count, location_text=None, latitude=None, longitude=None, source="Live Camera"):
    try:
        sender_email = EMAIL_CONFIG["sender_email"]
        sender_password = EMAIL_CONFIG["sender_password"]
        admin_email = EMAIL_CONFIG["admin_email"]
        if not sender_email or not sender_password or not admin_email:
            print("Email alert skipped: configure ALERT_EMAIL, ALERT_PASSWORD and ADMIN_EMAIL in Streamlit Secrets.")
            return False

        final_location = location_text
        if not final_location and latitude is not None and longitude is not None:
            final_location = f"Latitude: {float(latitude):.6f}\nLongitude: {float(longitude):.6f}"
        if not final_location:
            final_location = "GPS location unavailable."

        with threshold_store["lock"]:
            threshold = threshold_store["threshold"]
            place = threshold_store["place"]
            area = threshold_store["area_m2"]

        message = EmailMessage()
        message["Subject"] = "🚨 HIGH CROWD ALERT - CrowdGuard AI"
        message["From"] = sender_email
        message["To"] = admin_email
        message.set_content(
            f"""HIGH CROWD ALERT
==============================

Source:
{source}

Detected People:
{people_count}

Risk Level:
HIGH

Monitoring Place:
{place}

Monitored Area:
{area:.1f} m²

Adaptive HIGH Threshold:
{threshold} people

Location:
{final_location}

Please verify the situation and take appropriate action.

CrowdGuard AI
"""
        )
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as server:
            server.login(sender_email, sender_password)
            server.send_message(message)
        return True
    except Exception as exc:
        print("EMAIL ALERT ERROR:", type(exc).__name__, str(exc))
        return False


def send_high_crowd_alert_once(people_count, source, alert_id=None):
    location_text = gps_store.get("location_text")
    latitude = gps_store.get("latitude")
    longitude = gps_store.get("longitude")

    if source == "Live Photo Capture":
        if st.session_state.live_photo_alert_sent:
            return False
        sent = send_crowd_alert(people_count, location_text, latitude, longitude, source)
        if sent:
            st.session_state.live_photo_alert_sent = True
        return sent

    if source == "Uploaded Image":
        if alert_id is not None and st.session_state.uploaded_image_alert_id == alert_id:
            return False
        sent = send_crowd_alert(people_count, location_text, latitude, longitude, source)
        if sent:
            st.session_state.uploaded_image_alert_id = alert_id
        return sent

    if source == "Uploaded Video":
        if alert_id is not None and st.session_state.uploaded_video_alert_id == alert_id:
            return False
        sent = send_crowd_alert(people_count, location_text, latitude, longitude, source)
        if sent:
            st.session_state.uploaded_video_alert_id = alert_id
        return sent

    return send_crowd_alert(people_count, location_text, latitude, longitude, source)


# ============================================================
# CAMERA CALLBACK
# ============================================================

def camera_frame_callback(frame: av.VideoFrame):
    image = frame.to_ndarray(format="bgr24")
    try:
        people_count = detect_people(image)
    except Exception as exc:
        print("YOLO camera error:", exc)
        return frame

    with threshold_store["lock"]:
        threshold = int(threshold_store["threshold"])
        place = threshold_store["place"]

    risk = get_risk(people_count, threshold)

    with frame_store["lock"]:
        frame_store["frame"] = image.copy()

    should_alert = False
    alert_location = gps_store.get("location_text")
    alert_latitude = gps_store.get("latitude")
    alert_longitude = gps_store.get("longitude")

    with camera_store["lock"]:
        camera_store["people"] = people_count
        camera_store["risk"] = risk
        if people_count >= threshold:
            camera_store["high_frames"] += 1
        else:
            camera_store["high_frames"] = 0
        now = time.time()
        if (
            camera_store["high_frames"] >= HIGH_FRAMES_REQUIRED
            and now - camera_store["last_alert"] >= ALERT_COOLDOWN_SECONDS
        ):
            camera_store["last_alert"] = now
            should_alert = True

    if should_alert:
        threading.Thread(
            target=send_crowd_alert,
            kwargs={
                "people_count": people_count,
                "location_text": alert_location,
                "latitude": alert_latitude,
                "longitude": alert_longitude,
                "source": "Live Camera",
            },
            daemon=True,
        ).start()

    color = risk_color(risk)
    cv2.putText(image, f"People: {people_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(image, f"Risk: {risk}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(image, f"Place: {place}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(image, f"HIGH >= {threshold}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.putText(
        image,
        "GPS: ON" if gps_store.get("latitude") is not None else "GPS: OFF",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0) if gps_store.get("latitude") is not None else (0, 0, 255),
        2,
    )
    if risk == "HIGH":
        cv2.putText(image, "HIGH CROWD ALERT!", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return av.VideoFrame.from_ndarray(image, format="bgr24")


# ============================================================
# WEBRTC RECORDING / LIVE CAMERA
# ============================================================

def out_recorder_factory():
    return MediaRecorder(str(record_file), format="flv")


st.markdown("---")
st.subheader("📷🎥 Live Camera")
st.write("Click START to activate the camera. Configure the monitoring place and area in the sidebar before starting.")

left, center, right = st.columns([1, 2, 1])
with center:
    try:
        ctx = webrtc_streamer(
            key="crowd-live-camera-final",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=camera_frame_callback,
            out_recorder_factory=out_recorder_factory,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        )
    except Exception as exc:
        st.error("❌ Camera could not be started.")
        st.code(str(exc))
        ctx = None

camera_running = ctx is not None and ctx.state.playing
if camera_running:
    st.success("🟢 Camera is LIVE. YOLO detection and recording are active.")
else:
    st.info("🔴 Camera is stopped. Click START above to activate it.")


# ============================================================
# LIVE PHOTO
# ============================================================

st.markdown("---")
st.subheader("📸 Live Photo Capture")
capture_photo = st.button("📸 Capture Current Image", key="capture_live_photo", disabled=not camera_running)

if capture_photo:
    st.session_state.live_photo_alert_sent = False
    with frame_store["lock"]:
        current_frame = frame_store["frame"].copy() if frame_store["frame"] is not None else None

    if current_frame is None:
        st.warning("⚠️ Camera is running, but no processed frame is ready yet. Wait 1–2 seconds and try again.")
    else:
        photo_people = camera_store.get("people", 0)
        photo_risk = camera_store.get("risk", "LOW")
        st.session_state.captured_photo = current_frame.copy()
        st.session_state.captured_photo_people = photo_people
        st.session_state.captured_photo_risk = photo_risk
        st.session_state.captured_photo_threshold = crowd_threshold
        st.success("📸 Photo captured successfully!")

if st.session_state.captured_photo is not None:
    st.subheader("🖼️ Captured Crowd Image")
    st.image(cv2.cvtColor(st.session_state.captured_photo, cv2.COLOR_BGR2RGB), caption="Live Captured Crowd", width=CAMERA_DISPLAY_WIDTH)
    p1, p2, p3 = st.columns(3)
    with p1:
        st.metric("👥 People Detected", st.session_state.captured_photo_people)
    with p2:
        st.metric("⚠️ Crowd Risk", st.session_state.captured_photo_risk)
    with p3:
        st.metric("🚨 HIGH Threshold", st.session_state.captured_photo_threshold)

    if st.session_state.captured_photo_risk == "HIGH":
        st.error(f"🚨 HIGH CROWD DETECTED! People: {st.session_state.captured_photo_people}")
        alert_sent = send_high_crowd_alert_once(st.session_state.captured_photo_people, "Live Photo Capture")
        if alert_sent:
            st.success("📧 HIGH crowd alert email sent successfully.")
        elif st.session_state.live_photo_alert_sent:
            st.info("📧 HIGH crowd alert was already sent for this captured photo.")
        else:
            st.warning("⚠️ HIGH crowd detected, but the email alert could not be sent.")
    elif st.session_state.captured_photo_risk == "MEDIUM":
        st.warning(f"⚠️ MEDIUM CROWD LEVEL. People: {st.session_state.captured_photo_people}")
    else:
        st.success(f"✅ Crowd level is LOW. People: {st.session_state.captured_photo_people}")

    ok, encoded_image = cv2.imencode(".jpg", st.session_state.captured_photo, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if ok:
        st.download_button("📥 Download Captured Photo", encoded_image.tobytes(), "live_crowd_photo.jpg", "image/jpeg", key="download_live_photo")


# ============================================================
# LIVE VIDEO
# ============================================================

st.markdown("---")
st.subheader("🎥 Live Video Capture")
if camera_running:
    st.info("🎥 Live video recording is active. Click STOP when you want to finish recording.")
elif record_file.exists() and record_file.stat().st_size > 0:
    video_bytes = record_file.read_bytes()
    st.success("✅ Live crowd video recording completed.")
    st.video(video_bytes)
    st.download_button("📥 Download Live Crowd Video", video_bytes, "live_crowd_video.flv", "video/x-flv", key="download_live_crowd_video")
else:
    st.info("📹 No recorded video yet. Start the camera, keep it running, then click STOP.")


# ============================================================
# LIVE STATUS
# ============================================================

st.markdown("---")
st.subheader("📊 Live Camera Status")
current_people = camera_store["people"]
current_risk = camera_store["risk"]
s1, s2, s3, s4 = st.columns(4)
with s1: st.metric("👥 People Detected", current_people)
with s2: st.metric("⚠️ Current Risk", current_risk)
with s3: st.metric("🚨 HIGH Threshold", crowd_threshold)
with s4: st.metric("📐 Area", f"{current_area:.0f} m²")
if current_risk == "HIGH":
    st.error("🚨 HIGH CROWD DETECTED!")


# ============================================================
# HIGH-ACCURACY GPS LOCATION
# ============================================================

# Browser geolocation normally returns the first available position.
# This helper instead uses watchPosition() with high accuracy enabled
# and keeps the best accuracy reported during a short collection window.
# It does NOT invent or modify coordinates.
HIGH_ACCURACY_GPS_JS = """
new Promise((resolve) => {
    if (!navigator.geolocation) {
        resolve({error: {code: 0, message: "Browser does not support geolocation."}});
        return;
    }

    navigator.geolocation.getCurrentPosition(
        (position) => {
            const c = position.coords;
            resolve({
                coords: {
                    latitude: c.latitude,
                    longitude: c.longitude,
                    accuracy: c.accuracy,
                    altitude: c.altitude,
                    altitudeAccuracy: c.altitudeAccuracy,
                    heading: c.heading,
                    speed: c.speed
                },
                timestamp: position.timestamp
            });
        },
        (error) => {
            resolve({
                error: {
                    code: error.code,
                    message: error.message
                }
            });
        },
        {
            enableHighAccuracy: true,
            timeout: 5000,
            maximumAge: 0
        }
    );
})
"""


if "gps_request_id" not in st.session_state:
    st.session_state.gps_request_id = 0
if "gps_request_active" not in st.session_state:
    st.session_state.gps_request_active = False

st.markdown("---")
st.subheader("📍 GPS Location")
st.write(
    "GPS is independent from the camera. Click GET GPS LOCATION and allow browser location permission. "
    "CrowdGuard AI requests high-accuracy, fresh location data and waits briefly for the device to provide the GPS fix."
)

# The button only changes state. The JS component is rendered outside the
# button branch so streamlit-js-eval can return its asynchronous result reliably.
if st.button("📍 GET GPS LOCATION", key="get_high_accuracy_gps"):
    st.session_state.gps_request_id += 1
    st.session_state.gps_request_active = True
    st.session_state.gps_error_message = None
    st.rerun()

if st.button("🗑️ Clear GPS Location", key="clear_gps_location"):
    gps_store["latitude"] = None
    gps_store["longitude"] = None
    gps_store["accuracy"] = None
    gps_store["location_text"] = None
    st.session_state.gps_inferred_place = None
    st.session_state.gps_request_active = False
    st.session_state.gps_error_message = None
    st.rerun()

# Request a fresh browser position only after the user presses the GPS button.
# A new component key is created for every request.
gps_result = None
if st.session_state.gps_request_active:
    gps_result = streamlit_js_eval(
        js_expressions=HIGH_ACCURACY_GPS_JS,
        want_output=True,
        key=f"HIGH_ACCURACY_GPS_{st.session_state.gps_request_id}",
    )

if st.session_state.gps_request_active and isinstance(gps_result, dict):
    st.session_state.gps_request_active = False

    if "error" in gps_result:
        error_info = gps_result.get("error") or {}
        code = error_info.get("code")
        message = error_info.get("message", "Unknown GPS error")
        st.session_state.gps_error_message = message

        if code == 1:
            st.error("❌ Browser location permission was denied. Check the browser site-location setting and try again.")
        elif code == 2:
            st.warning(f"⚠️ Location is temporarily unavailable: {message}")
        elif code == 3:
            st.warning("⚠️ GPS request timed out before a usable position was received. Try again outdoors or near a window.")
        else:
            st.warning(f"⚠️ GPS error: {message}")
    elif gps_result.get("coords"):
        coords = gps_result["coords"]
        lat = coords.get("latitude")
        lon = coords.get("longitude")
        accuracy = coords.get("accuracy")

        if lat is not None and lon is not None and accuracy is not None:
            lat = float(lat)
            lon = float(lon)
            accuracy = float(accuracy)

            # Do not accept a coarse location. This prevents a wrong city
            # from being displayed or used in crowd alerts.
            MAX_ACCEPTED_GPS_ACCURACY = 1000.0

            if accuracy <= MAX_ACCEPTED_GPS_ACCURACY:
                gps_store["latitude"] = lat
                gps_store["longitude"] = lon
                gps_store["accuracy"] = accuracy
                gps_store["location_text"] = reverse_geocode(lat, lon)
                inferred = infer_place_from_text(gps_store["location_text"] or "")
                st.session_state.gps_inferred_place = inferred
                st.session_state.gps_error_message = None
            else:
                # Explicitly reject coarse browser/IP/Wi-Fi results.
                gps_store["latitude"] = None
                gps_store["longitude"] = None
                gps_store["accuracy"] = accuracy
                gps_store["location_text"] = None
                st.session_state.gps_inferred_place = None
                st.session_state.gps_error_message = (
                    f"GPS result rejected because its accuracy is too low (±{accuracy:.0f} m)."
                )
        else:
            st.session_state.gps_error_message = "The browser returned an incomplete GPS result."

    # Force one clean rerun after the asynchronous component result has been processed.
    st.rerun()

if st.session_state.get("gps_error_message"):
    message = st.session_state.gps_error_message
    if "accuracy is too low" in message:
        st.error(f"❌ {message}")
        st.warning(
            "The browser returned a coarse location, so CrowdGuard AI will not show it on the map "
            "or use it for alerts. High-accuracy mode is already enabled. On a laptop, the device "
            "may still only provide Wi-Fi/IP-based positioning. A phone with Location Services/GPS "
            "enabled normally has a better chance of providing a precise fix."
        )
    else:
        st.warning(f"⚠️ {message}")

if st.session_state.gps_request_active:
    st.info(
        "📡 Searching for the best high-accuracy location for up to 30 seconds. "
        "Please keep the page open while the device improves the fix..."
    )

latitude = gps_store["latitude"]
longitude = gps_store["longitude"]
accuracy = gps_store["accuracy"]
location_text = gps_store["location_text"]

if latitude is not None and longitude is not None:
    st.success("📍 Accurate GPS location accepted.")
    if location_text:
        st.info(f"📍 **Location:** {location_text}")
    else:
        st.warning("📍 GPS coordinates were accepted, but the readable location name could not be determined.")
        st.code(f"Latitude: {latitude:.6f}\nLongitude: {longitude:.6f}", language="text")

    if st.session_state.gps_inferred_place:
        st.info(
            f"🤖 GPS text suggests: **{st.session_state.gps_inferred_place}**. "
            "This is only a suggestion; manual place selection is recommended when the exact room/area is known."
        )

    if accuracy is not None:
        st.caption(f"GPS accuracy: approximately ±{accuracy:.1f} m")

    if not location_text:
        st.caption("Tip: reverse geocoding depends on internet/service availability; GPS coordinates remain valid when the address lookup is unavailable.")

    gps_map = folium.Map(location=[latitude, longitude], zoom_start=16)
    folium.Marker(
        [latitude, longitude],
        popup=location_text or f"Current Location\n{latitude:.6f}, {longitude:.6f}",
        tooltip="📍 Crowd Monitoring Location",
    ).add_to(gps_map)
    st_folium(gps_map, width=700, height=350, key="current_gps_map")
else:
    st.info("📍 No accurate GPS location is currently accepted. Click GET GPS LOCATION and allow permission.")


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown("---")
st.subheader("📁 Upload Crowd Image")
image_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="crowd_image_upload")

if image_file is not None:
    image_alert_id = f"{image_file.name}_{image_file.size}"
    image_array = np.frombuffer(image_file.getvalue(), dtype=np.uint8)
    upload_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if upload_image is None:
        st.error("❌ Unable to read image.")
    else:
        people_count = detect_people(upload_image, conf=0.15, imgsz=960, iou=0.50, max_det=1000)
        risk = get_risk(people_count, crowd_threshold)
        st.subheader("📷 Image Crowd Analysis")
        st.image(cv2.cvtColor(upload_image, cv2.COLOR_BGR2RGB), caption="Detected Crowd", width=650)
        u1, u2, u3 = st.columns(3)
        with u1: st.metric("👥 People Detected", people_count)
        with u2: st.metric("⚠️ Crowd Risk", risk)
        with u3: st.metric("🚨 HIGH Threshold", crowd_threshold)

        if risk == "HIGH":
            st.error(f"🚨 HIGH CROWD DETECTED! People: {people_count}")
            st.warning(f"📍 Alert Location: {location_text or 'GPS unavailable'}")
            if st.session_state.uploaded_image_alert_id != image_alert_id:
                alert_sent = send_high_crowd_alert_once(people_count, "Uploaded Image", image_alert_id)
                if alert_sent:
                    st.success("📧 HIGH crowd alert email sent successfully.")
                else:
                    st.warning("⚠️ HIGH crowd detected, but the email alert could not be sent.")
                    st.info("Check Gmail/Streamlit Secrets configuration if email alerts are required.")
            else:
                st.info("📧 Alert already sent for this uploaded image.")
        elif risk == "MEDIUM":
            st.warning(f"⚠️ MEDIUM CROWD LEVEL. People: {people_count}")
        else:
            st.success(f"✅ LOW CROWD LEVEL. People: {people_count}")


# ============================================================
# VIDEO UPLOAD
# ============================================================

st.markdown("---")
st.subheader("🎥 Upload Crowd Video")
uploaded_video = st.file_uploader("Choose a video", type=["mp4", "avi", "mov", "mkv"], key="crowd_video_upload")

if uploaded_video is not None:
    video_alert_id = f"{uploaded_video.name}_{uploaded_video.size}"
    extension = Path(uploaded_video.name).suffix.lower() or ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as input_file:
        input_file.write(uploaded_video.getvalue())
        input_path = input_file.name

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error("❌ Unable to open video.")
        try:
            os.remove(input_path)
        except OSError:
            pass
    else:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip = 2

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        time_data = []
        people_data = []
        max_people = 0
        frame_number = 0
        last_people_count = 0

        progress = st.progress(0)
        status = st.empty()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_number % frame_skip == 0:
                    people_count = detect_people(frame, conf=0.15, imgsz=640, iou=0.50, max_det=1000)
                    last_people_count = people_count
                else:
                    people_count = last_people_count

                max_people = max(max_people, people_count)
                risk = get_risk(people_count, crowd_threshold)
                color = risk_color(risk)
                cv2.putText(frame, f"People: {people_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, f"Risk: {risk}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, f"Place: {current_place}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(frame, f"HIGH >= {crowd_threshold}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                if risk == "HIGH":
                    cv2.putText(frame, "HIGH CROWD ALERT!", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                out.write(frame)
                video_time = frame_number / fps
                time_data.append(round(video_time, 2))
                people_data.append(people_count)
                frame_number += 1

                if total_frames > 0:
                    percentage = min(frame_number / total_frames, 1.0)
                    progress.progress(percentage)
                    status.write(f"Processing video... {int(percentage * 100)}% | People: {people_count} | Risk: {risk}")
        finally:
            cap.release()
            out.release()

        progress.empty()
        status.empty()

        st.success("🎉 Video processing completed!")
        with open(output_path, "rb") as video_file:
            processed_video = video_file.read()
        st.subheader("🎬 Processed Video")
        st.video(processed_video)

        final_risk = get_risk(max_people, crowd_threshold)
        st.subheader("📊 Video Crowd Analysis")
        v1, v2, v3 = st.columns(3)
        with v1: st.metric("👥 Maximum People Detected", max_people)
        with v2: st.metric("⚠️ Maximum Crowd Risk", final_risk)
        with v3: st.metric("🚨 HIGH Threshold", crowd_threshold)

        if final_risk == "HIGH":
            st.error(f"🚨 HIGH CROWD DETECTED! Maximum people: {max_people}")
            st.warning(f"📍 Alert Location: {location_text or 'GPS unavailable'}")
            if st.session_state.uploaded_video_alert_id != video_alert_id:
                alert_sent = send_high_crowd_alert_once(max_people, "Uploaded Video", video_alert_id)
                if alert_sent:
                    st.success("📧 HIGH crowd alert email sent successfully.")
                else:
                    st.warning("⚠️ HIGH crowd detected, but the email alert could not be sent.")
                    st.info("Check Gmail/Streamlit Secrets configuration if email alerts are required.")
            else:
                st.info("📧 Alert already sent for this uploaded video.")
        elif final_risk == "MEDIUM":
            st.warning(f"⚠️ MEDIUM CROWD LEVEL. Maximum people: {max_people}")
        else:
            st.success(f"✅ LOW CROWD LEVEL. Maximum people: {max_people}")

        if time_data:
            graph_df = pd.DataFrame({"Time (seconds)": time_data, "People": people_data})
            st.subheader("📈 Crowd Count Over Time")
            st.line_chart(graph_df, x="Time (seconds)", y="People")

        st.download_button(
            "📥 Download Processed Video",
            processed_video,
            "crowd_analysis_output.mp4",
            "video/mp4",
            key="download_uploaded_processed_video",
        )

        csv_df = pd.DataFrame({"Time (seconds)": time_data, "People Detected": people_data})
        st.download_button(
            "📄 Download Crowd Report",
            csv_df.to_csv(index=False),
            "crowd_analysis_report.csv",
            "text/csv",
            key="download_uploaded_crowd_report",
        )

        for temporary_path in [input_path, output_path]:
            try:
                os.remove(temporary_path)
            except OSError:
                pass


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "CrowdGuard AI • YOLOv8 • Adaptive Place Threshold • Live Photo • Live Video • "
    "GPS • Location Text • Image Upload • Video Upload • HIGH-Crowd Email Alerts"
)


# ============================================================
# CROWDGUARD AI - ADDITIONAL ANALYTICS FEATURES
# NOTE: This section is ADDITIVE. The original CrowdGuard AI
# features above are kept unchanged.
# ============================================================

if "crowd_history" not in st.session_state:
    st.session_state.crowd_history = []
if "feature_session_start" not in st.session_state:
    st.session_state.feature_session_start = time.time()
if "last_feature_snapshot" not in st.session_state:
    st.session_state.last_feature_snapshot = 0.0
if "auto_evidence_path" not in st.session_state:
    st.session_state.auto_evidence_path = None
if "high_event_start" not in st.session_state:
    st.session_state.high_event_start = None
if "last_high_event_duration" not in st.session_state:
    st.session_state.last_high_event_duration = 0.0


def add_crowd_history_point():
    """Store the latest live-camera observation for trend analytics."""
    with camera_store["lock"]:
        people = int(camera_store["people"])
        risk = str(camera_store["risk"])

    with threshold_store["lock"]:
        threshold = int(threshold_store["threshold"])
        area = float(threshold_store["area_m2"])
        place = str(threshold_store["place"])

    density = people / area if area > 0 else 0.0
    now = time.time()

    # Avoid filling the history with duplicate observations when the page
    # is refreshed repeatedly without a new camera observation.
    if st.session_state.crowd_history:
        previous = st.session_state.crowd_history[-1]
        if (
            previous["people"] == people
            and previous["risk"] == risk
            and now - previous["timestamp"] < 1.0
        ):
            return

    st.session_state.crowd_history.append(
        {
            "timestamp": now,
            "time": datetime.fromtimestamp(now).strftime("%H:%M:%S"),
            "people": people,
            "risk": risk,
            "threshold": threshold,
            "density": density,
            "place": place,
        }
    )

    # Keep the browser session lightweight.
    if len(st.session_state.crowd_history) > 500:
        st.session_state.crowd_history = st.session_state.crowd_history[-500:]


def save_current_evidence(reason="HIGH crowd event"):
    """Save the latest processed camera frame as evidence."""
    with frame_store["lock"]:
        frame = None if frame_store["frame"] is None else frame_store["frame"].copy()

    if frame is None:
        return None

    EVIDENCE_DIR = BASE_DIR / "crowd_evidence"
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    filename = EVIDENCE_DIR / f"crowdguard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.putText(
        frame,
        f"Evidence: {reason}",
        (20, max(30, frame.shape[0] - 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
    )

    if cv2.imwrite(str(filename), frame):
        return filename
    return None


# datetime is imported here so the original import section remains untouched.
from datetime import datetime

@st.fragment(run_every="30s")
def render_advanced_crowd_analytics():
    """Refresh crowd analytics automatically every 30 seconds."""
    st.markdown("---")
    st.subheader("🧠 Advanced Crowd Analytics")
    st.caption(
        "These additional analytics use observations collected from the existing "
        "single-camera CrowdGuard AI session. Prediction is an experimental project feature, "
        "not an official safety forecast."
    )

    refresh_col, clear_col = st.columns([3, 1])
    with refresh_col:
        if st.button("🔄 Update Crowd Analytics", use_container_width=True, key="update_crowd_analytics"):
            add_crowd_history_point()
            st.rerun()
    with clear_col:
        if st.button("🗑️ Clear History", use_container_width=True, key="clear_crowd_history"):
            st.session_state.crowd_history = []
            st.session_state.feature_session_start = time.time()
            st.session_state.auto_evidence_path = None
            st.session_state.high_event_start = None
            st.session_state.last_high_event_duration = 0.0
            st.rerun()

    # Automatically save one observation whenever the page is opened/refreshed
    # after the camera has produced a meaningful observation.
    with camera_store["lock"]:
        latest_people = int(camera_store["people"])
        latest_risk = str(camera_store["risk"])

    if latest_people > 0:
        add_crowd_history_point()

    history = st.session_state.crowd_history

    if history:
        history_df = pd.DataFrame(history)
        history_df["datetime"] = pd.to_datetime(history_df["timestamp"], unit="s")

        current_people = int(history_df.iloc[-1]["people"])
        max_people = int(history_df["people"].max())
        average_people = float(history_df["people"].mean())
        current_density = float(history_df.iloc[-1]["density"])
        high_events = int((history_df["risk"] == "HIGH").sum())

        # Trend uses the difference between recent observations.
        if len(history_df) >= 2:
            recent = history_df["people"].tail(min(5, len(history_df))).to_numpy(dtype=float)
            trend_change = float(recent[-1] - recent[0])
            if trend_change > 1:
                trend_label = "↗ Increasing"
            elif trend_change < -1:
                trend_label = "↘ Decreasing"
            else:
                trend_label = "→ Stable"
        else:
            trend_change = 0.0
            trend_label = "→ Waiting for more observations"

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("👥 Current People", current_people)
        with m2:
            st.metric("📈 Maximum People", max_people)
        with m3:
            st.metric("📐 Current Density", f"{current_density:.2f}/m²")
        with m4:
            st.metric("🚨 HIGH Observations", high_events)

        st.markdown("### 📈 Crowd Trend")
        trend_chart_df = history_df[["datetime", "people"]].copy()
        trend_chart_df = trend_chart_df.set_index("datetime")
        st.line_chart(trend_chart_df, y="people", use_container_width=True)

        trend_col1, trend_col2 = st.columns(2)
        with trend_col1:
            st.info(f"**Current trend:** {trend_label}")
            st.caption(f"Change across recent observations: {trend_change:+.0f} people")
        with trend_col2:
            session_minutes = max(
                0.0,
                (history_df["timestamp"].iloc[-1] - history_df["timestamp"].iloc[0]) / 60.0,
            )
            st.info(f"**Monitoring observations:** {len(history_df)}")
            st.caption(f"Observed session span: {session_minutes:.1f} minutes")

        # --------------------------------------------------------
        # SHORT-TERM EXPERIMENTAL PREDICTION
        # --------------------------------------------------------
        st.markdown("### 🔮 Short-Term Crowd Trend Prediction")

        if len(history_df) >= 4:
            x = (history_df["timestamp"] - history_df["timestamp"].iloc[0]).to_numpy(dtype=float)
            y = history_df["people"].to_numpy(dtype=float)
            x_recent = x[-min(20, len(x)):]
            y_recent = y[-min(20, len(y)):]

            if len(np.unique(x_recent)) >= 2:
                slope, intercept = np.polyfit(x_recent, y_recent, 1)
                # Project a short 30-second window from the latest observation.
                prediction_seconds = 30.0
                predicted_people = max(
                    0,
                    int(round(y_recent[-1] + slope * prediction_seconds)),
                )
                predicted_risk = get_risk(predicted_people, crowd_threshold)

                p1, p2, p3 = st.columns(3)
                with p1:
                    st.metric("Current", f"{current_people} people")
                with p2:
                    st.metric("Predicted (~30 sec)", f"{predicted_people} people")
                with p3:
                    st.metric("Predicted Risk", predicted_risk)

                if slope > 0.02:
                    st.warning("⚠️ The recent crowd trend is increasing.")
                elif slope < -0.02:
                    st.success("✅ The recent crowd trend is decreasing.")
                else:
                    st.info("ℹ️ The recent crowd trend is relatively stable.")
            else:
                st.info("Collect a few more observations for a prediction.")
        else:
            st.info("Collect at least 4 camera observations to enable the experimental prediction.")

        # --------------------------------------------------------
        # CROWD DURATION
        # --------------------------------------------------------
        st.markdown("### ⏱️ HIGH-Crowd Duration")

        high_mask = history_df["risk"] == "HIGH"
        if high_mask.any():
            high_times = history_df.loc[high_mask, "timestamp"].to_numpy(dtype=float)
            duration_seconds = float(max(0.0, high_times[-1] - high_times[0]))
            st.metric("Observed HIGH duration", f"{duration_seconds:.0f} seconds")
            st.caption("This is the duration covered by recorded HIGH observations in the current browser session.")
        else:
            st.success("No HIGH-crowd observations recorded in the current session.")

        # --------------------------------------------------------
        # AUTOMATIC EVIDENCE SNAPSHOT
        # --------------------------------------------------------
        st.markdown("### 📸 Crowd Evidence")

        if latest_risk == "HIGH" and st.session_state.auto_evidence_path is None:
            evidence_path = save_current_evidence("HIGH crowd detected")
            if evidence_path is not None:
                st.session_state.auto_evidence_path = str(evidence_path)

        if st.session_state.auto_evidence_path:
            evidence_path = Path(st.session_state.auto_evidence_path)
            if evidence_path.exists():
                st.image(str(evidence_path), caption="Latest HIGH-crowd evidence", width=650)
                with open(evidence_path, "rb") as evidence_file:
                    st.download_button(
                        "📥 Download Evidence Snapshot",
                        evidence_file.read(),
                        evidence_path.name,
                        "image/jpeg",
                        key="download_crowd_evidence",
                    )
        else:
            st.info("A snapshot will be saved when a HIGH-crowd observation is available.")

        # --------------------------------------------------------
        # SESSION REPORT
        # --------------------------------------------------------
        st.markdown("### 📝 Monitoring Session Report")

        report_df = history_df[
            ["datetime", "place", "people", "density", "threshold", "risk"]
        ].copy()
        report_df.columns = [
            "Time",
            "Monitoring Place",
            "People Detected",
            "Density (people/m²)",
            "HIGH Threshold",
            "Risk",
        ]

        report_text = (
            "CROWDGUARD AI - MONITORING SESSION REPORT\n"
            "==========================================\n\n"
            f"Monitoring Place: {current_place}\n"
            f"Monitored Area: {current_area:.1f} m²\n"
            f"Density Limit: {current_density:.2f} people/m²\n"
            f"HIGH Threshold: {crowd_threshold} people\n"
            f"Current People: {current_people}\n"
            f"Maximum People: {max_people}\n"
            f"Average People: {average_people:.2f}\n"
            f"Current Density: {current_density:.2f} people/m²\n"
            f"HIGH Observations: {high_events}\n"
            f"Trend: {trend_label}\n\n"
            "Observation Details\n"
            "--------------------\n"
            f"{report_df.to_string(index=False)}\n"
        )

        st.download_button(
            "📄 Download Session Report",
            report_text,
            "crowdguard_session_report.txt",
            "text/plain",
            key="download_crowdguard_session_report",
        )

        st.download_button(
            "📊 Download Analytics CSV",
            report_df.to_csv(index=False),
            "crowdguard_analytics.csv",
            "text/csv",
            key="download_crowdguard_analytics_csv",
        )
    else:
        st.info("Start the live camera and allow it to detect people. CrowdGuard AI will then build the analytics history here.")


render_advanced_crowd_analytics()
