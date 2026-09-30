import streamlit as st
import base64
import os


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CROWDGURD AI-Intelligent Crowd Management System",
    page_icon="👥",
    layout="wide"
)


# =========================================================
# BACKGROUND
# =========================================================

def set_background(image_path):

    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(
            image_file.read()
        ).decode()

    st.markdown(
        f"""
        <style>

        .stApp {{
            background-image:
                linear-gradient(
                    rgba(0, 0, 20, 0.55),
                    rgba(0, 0, 20, 0.55)
                ),
                url("data:image/jpeg;base64,{encoded}");

            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .main .block-container {{
            color: white !important;
        }}

        .main .block-container h1,
        .main .block-container h2,
        .main .block-container h3,
        .main .block-container h4,
        .main .block-container h5,
        .main .block-container h6,
        .main .block-container p,
        .main .block-container li,
        .main .block-container span {{
            color: white !important;
        }}

        section[data-testid="stSidebar"] {{
            background-color: rgba(3, 12, 35, 0.96);
        }}

        section[data-testid="stSidebar"] * {{
            color: white !important;
        }}

        .stButton > button {{
            color: white !important;
            background-color: #0878d1 !important;
            border: 1px solid #4db8ff !important;
            border-radius: 10px;
            font-weight: bold;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )


# Apply background
background_path = "image/crowdguard_background.jpg"

if os.path.exists(background_path):
    set_background(background_path)


# =========================================================
# LOGO
# =========================================================

logo_path = os.path.join("image", "logo.jpeg")

if os.path.exists(logo_path):

    col1, col2, col3 = st.columns([3, 2, 3])

    with col2:

        st.image(
            logo_path,
            width=600
        )

else:

    st.warning("Logo image not found.")


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🤖 Artificial Intelligence Careers for Women (AICW)")

    st.title(" 🎓 Capstone Project")

    st.markdown("---")

    st.markdown("### 👩‍💻 Team Members")

    st.write(
        "1. D.L. Priyanka  \n"
        "doddipatlapriyanka1@gmail.com"
    )

    st.write(
        "2. R.S.V. Madhulika  \n"
        "rayudusrividyamadhulika@gmail.com"
    )

    st.write(
        "3. Y. Leela Devi  \n"
        "leelayejarla@gmail.com"
    )

    st.write(
        "4. K. Vyjayanthi  \n"
        "vyshu366@gmail.com"
    )

    st.markdown("---")

    st.markdown("### 🏫 College Name")

    st.write(
        "VSM COLLEGE OF ENGINEERING"
    )

    st.write(
        "Ramachandrapuram"
    )

    st.markdown("---")

    st.markdown("### 👨‍🏫 Guide Name")

    st.write("Abdul Aziz Md, Lead - AICW (South)")


# =========================================================
# MAIN PAGE
# =========================================================

st.markdown(
    """
    <h1 style="text-align:center;">
        👥 CROWDGURD AI-Intelligent Crowd Management System
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h3 style="text-align:center;">
        AI-Powered Crowd Detection, Counting and Risk Analysis
    </h3>
    """,
    unsafe_allow_html=True
)

st.markdown("---")


# =========================================================
# PROJECT TITLE
# =========================================================

st.header("🚨 CROWDGURD AI-Intelligent Crowd Management System")

st.write(
    """
    An intelligent computer vision-based application designed to
    monitor crowds, detect people, count individuals and identify
    different levels of crowd risk.
    """
)


# =========================================================
# OPEN PROJECT APPLICATION
# =========================================================

st.subheader("🚀 Project Application")

st.write(
    """
    The complete AI Crowd Management application is available
    on the next page. Click the button below to continue.
    """
)

# IMPORTANT:
# This opens the second page in the SAME localhost application.
# It does NOT use the deployed Streamlit URL.

if st.button(
    "🛡️ OPEN CROWDGUARD AI APPLICATION",
    type="primary",
    use_container_width=True
):

    st.switch_page(
        "pages/1_Crowd_Management_Application.py"
    )


st.markdown("---")


# =========================================================
# PROJECT DESCRIPTION
# =========================================================

st.header("📋 Project Description")

st.write(
    """
    The CrowdGurd AI is an intelligent Crowd Management System uses Artificial Intelligence
    and Computer Vision to analyze crowd situations.

    The system uses the YOLO object detection model to detect
    people from images, videos and live camera input.

    It automatically counts the detected people and classifies
    the crowd into different risk levels:

    • LOW  
    • MEDIUM  
    • HIGH

    The system is designed to support crowd monitoring and
    early identification of potentially crowded situations.
    """
)


# =========================================================
# NEW FEATURE
# AUTOMATIC CROWD ANALYSIS
# =========================================================

st.header("🤖 Automatic Crowd Analysis")

st.write(
    """
    CrowdGurd AI can use the monitored place and area to
    automatically calculate a suitable HIGH crowd threshold.

    Instead of requiring the user to manually enter a crowd
    threshold, the application uses the monitored area and
    a defined crowd-density limit.
    """
)

auto_col1, auto_col2, auto_col3 = st.columns(3)

with auto_col1:

    st.info(
        """
        ### 📍 1. Identify Place

        The monitoring area can be selected
        according to the type of location.

        Examples:

        • Classroom  
        • Laboratory  
        • Corridor  
        • Auditorium  
        • Canteen  
        • Outdoor Area
        """
    )

with auto_col2:

    st.info(
        """
        ### 📐 2. Determine Area

        The monitored area is specified
        in square meters.

        Example:

        **50 m² classroom**
        """
    )

with auto_col3:

    st.info(
        """
        ### 👥 3. Calculate Threshold

        The system calculates the HIGH
        crowd threshold automatically.

        **Area × Density Limit**
        """
    )


# =========================================================
# AI MONITORING WORKFLOW
# =========================================================

st.header("🧠 AI Monitoring Workflow")

workflow1, workflow2, workflow3, workflow4 = st.columns(4)

with workflow1:

    st.markdown(
        """
        ### 📷 Step 1

        **Capture / Upload**

        Capture a place image or upload
        an image/video, or use the live
        camera.
        """
    )

with workflow2:

    st.markdown(
        """
        ### 🤖 Step 2

        **AI Detection**

        YOLO detects people in the
        monitored scene.
        """
    )

with workflow3:

    st.markdown(
        """
        ### 👥 Step 3

        **People Counting**

        The detected people are counted
        and compared with the calculated
        crowd threshold.
        """
    )

with workflow4:

    st.markdown(
        """
        ### 🚨 Step 4

        **Risk Analysis**

        The crowd is classified into:

        🟢 LOW  
        🟡 MEDIUM  
        🔴 HIGH
        """
    )


st.markdown("---")


# =========================================================
# AUTOMATIC THRESHOLD
# =========================================================

st.header("📐 Automatic Crowd Threshold")

st.write(
    """
    The automatic threshold is calculated using the monitored
    area and the defined crowd-density limit for the selected
    type of location.
    """
)

formula_col1, formula_col2 = st.columns(2)

with formula_col1:

    st.markdown(
        """
        ### 🧮 Calculation

        **Automatic HIGH Threshold**

        **Monitored Area × Density Limit**

        Example:

        **50 m² × 1.50 people/m² = 75 people**

        Therefore:

        🚨 **HIGH Threshold = 75 people**
        """
    )

with formula_col2:

    st.success(
        """
        ### 🤖 Automatic Setting

        No manual HIGH threshold is required.

        The application can calculate the threshold
        according to the monitored location and area.

        Different places can therefore have different
        crowd limits.
        """
    )


# =========================================================
# RISK LEVELS
# =========================================================

st.header("🚨 Crowd Risk Levels")

risk1, risk2, risk3 = st.columns(3)

with risk1:

    st.success(
        """
        ### 🟢 LOW

        The detected crowd is within
        the lower range of the calculated
        threshold.
        """
    )

with risk2:

    st.warning(
        """
        ### 🟡 MEDIUM

        The detected crowd is approaching
        the calculated HIGH threshold.
        """
    )

with risk3:

    st.error(
        """
        ### 🔴 HIGH

        The detected crowd reaches or
        exceeds the calculated HIGH
        threshold.

        An alert can be generated.
        """
    )


# =========================================================
# MONITORING MODES
# =========================================================

st.header("📊 Monitoring Modes")

mode1, mode2, mode3 = st.columns(3)

with mode1:

    st.markdown(
        """
        ### 📷 Image Analysis

        • Upload crowd image  
        • Detect people using AI  
        • Count detected people  
        • Analyze crowd risk
        """
    )

with mode2:

    st.markdown(
        """
        ### 🎥 Video Analysis

        • Upload crowd video  
        • Analyze video frames  
        • Detect and count people  
        • Analyze crowd changes
        """
    )

with mode3:

    st.markdown(
        """
        ### 📹 Live Camera

        • Access live camera  
        • Real-time detection  
        • Live people counting  
        • HIGH crowd monitoring
        """
    )


st.markdown("---")


# =========================================================
# SMART ALERT SYSTEM
# =========================================================

st.header("🚨 Smart Alert System")

alert_col1, alert_col2 = st.columns(2)

with alert_col1:

    st.info(
        """
        ### 🔔 Automatic Crowd Alert

        When the detected crowd reaches the
        HIGH threshold, the system can generate
        an alert for the monitored situation.
        """
    )

with alert_col2:

    st.info(
        """
        ### 📍 Location Information

        GPS and monitoring-location information
        can be associated with the crowd alert
        to help identify where the monitoring
        occurred.
        """
    )


# =========================================================
# SYSTEM ARCHITECTURE
# =========================================================

st.header("🏗️ System Architecture")

st.markdown(
    """
    <div style="
        padding:25px;
        margin-top:10px;
        margin-bottom:20px;
        border-radius:15px;
        background:rgba(0,0,0,0.45);
        text-align:center;
        font-size:18px;
        line-height:2;
        border:1px solid rgba(255,255,255,0.2);
    ">

    📷 <b>Image / Video / Live Camera</b>

    <br>↓<br>

    🤖 <b>YOLO AI Detection</b>

    <br>↓<br>

    👥 <b>People Counting</b>

    <br>↓<br>

    📐 <b>Area + Crowd Density</b>

    <br>↓<br>

    🚨 <b>Automatic HIGH Threshold</b>

    <br>↓<br>

    📊 <b>Risk Analysis</b>

    <br>↓<br>

    🟢 <b>LOW</b>
    &nbsp;&nbsp;&nbsp;
    🟡 <b>MEDIUM</b>
    &nbsp;&nbsp;&nbsp;
    🔴 <b>HIGH</b>

    <br>↓<br>

    🔔 <b>Alert & Location Information</b>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# KEY FEATURES
# =========================================================

st.header("✨ Key Features")

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        """
        ### 📷 Image Analysis

        • Upload crowd images  
        • Detect people using AI  
        • Count detected people  
        • Display crowd risk level
        """
    )

with col2:

    st.markdown(
        """
        ### 🎥 Video & Live Camera

        • Analyze crowd videos  
        • Live camera monitoring  
        • Real-time people detection  
        • People counting
        """
    )


col3, col4 = st.columns(2)

with col3:

    st.markdown(
        """
        ### 🚨 Risk Detection

        • LOW crowd level  
        • MEDIUM crowd level  
        • HIGH crowd level  
        • High-crowd alert
        """
    )

with col4:

    st.markdown(
        """
        ### 📍 Location Monitoring

        • GPS location support  
        • Crowd monitoring location  
        • Location information  
        • Alert location information
        """
    )


st.markdown("---")


# =========================================================
# PROJECT GOAL
# =========================================================

st.header("🎯 Project Goal")

st.info(
    """
    To provide an AI-powered solution for monitoring crowds,
    identifying potentially high-risk crowd situations and
    supporting safer public spaces.
    """
)


# =========================================================
# TECHNOLOGIES
# =========================================================

st.header("🛠️ Technologies Used")

tech1, tech2, tech3, tech4 = st.columns(4)

with tech1:

    st.info("🐍 Python")

with tech2:

    st.info("🤖 YOLO")

with tech3:

    st.info("👁️ Computer Vision")

with tech4:

    st.info("🌐 Streamlit")


st.markdown("---")


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <h4 style="text-align:center;">
        Artificial Intelligence Careers for Women (AICW)
    </h4>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="text-align:center;">
        Capstone Project • VSM College of Engineering
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="text-align:center;">
        Guided by Abdul Aziz Md, Lead - AICW (South)
    </p>
    """,
    unsafe_allow_html=True
)
