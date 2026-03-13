import streamlit as st
import cv2
import numpy as np
from PIL import Image
import base64
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Image - Global and Adaptive Thresholding",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- STYLE AND CONSTANTS ---
BRAND_COLOR = "#1E88E5"
ACCENT_COLOR = "#7C4DFF"
BG_COLOR = "#0E1117"

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_faded_bg(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    page_bg_img = '''
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: linear-gradient(rgba(14, 17, 23, 0.4), rgba(14, 17, 23, 0.5)), url("data:image/png;base64,%s");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Apply background
script_dir = os.path.dirname(os.path.abspath(__file__))
BG_IMG_PATH = os.path.join(script_dir, "orion_nasa.png")
if os.path.exists(BG_IMG_PATH):
    set_faded_bg(BG_IMG_PATH)

# Custom CSS for Glassmorphism and modern look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    /* General text default */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3em;
        background-color: #1E88E5;
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
    }

    .stButton>button:hover {
        background-color: #7C4DFF;
        box-shadow: 0 4px 15px rgba(124, 77, 255, 0.4);
        transform: translateY(-2px);
    }

    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
    }

    /* Target labels and specific markdown text to be white for readability in the MAIN area */
    [data-testid="stAppViewContainer"] label, 
    [data-testid="stAppViewContainer"] p, 
    [data-testid="stAppViewContainer"] h3, 
    [data-testid="stAppViewContainer"] h4, 
    [data-testid="stAppViewContainer"] h5, 
    [data-testid="stAppViewContainer"] .stMetric div {
        color: white !important;
    }

    /* Target generic spans in main area but EXCLUDE headers */
    [data-testid="stAppViewContainer"] span:not(h1 span):not(h2 span) {
        color: white;
    }

    /* SIDEBAR SPECIFIC: Make these black for visibility on light blurred background */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] span {
        color: black !important;
    }

    /* FORCE Light Blue on all header levels and their nested children */
    h1, h1 span, h2, h2 span, .welcome-text {
        color: #007ACC !important;
        font-weight: 800 !important;
    }

    /* Ensure specific Streamlit components in sidebar use black labels */
    [data-testid="stSidebar"] .stSlider label, 
    [data-testid="stSidebar"] .stSelectbox label, 
    [data-testid="stSidebar"] .stRadio label, 
    [data-testid="stSidebar"] .stFileUploader label {
        color: black !important;
    }

    .css-1d391kg {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    .main {
        background: transparent;
    }

    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
    }
    
    .stSlider > div > div > div > div {
        color: #1E88E5;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px 10px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: white;
        border: 1px solid transparent;
        transition: all 0.2s;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(30, 136, 229, 0.1);
        border-bottom: 2px solid #1E88E5 !important;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    # Attribution in the upper left (sidebar top)
    st.sidebar.markdown(
        "<p style='font-size: 0.8rem; color: white; margin-bottom: 0;'>courtesy of: anaofeliamt.ai</p>", 
        unsafe_allow_html=True
    )
    st.sidebar.markdown("---")

    # Main Title with forced color
    st.markdown(
        "<h1 style='color: #007ACC !important; font-weight: 800; letter-spacing: -1px; margin-bottom: 0px; padding-bottom: 0px;'>Image Processing Studio</h1>", 
        unsafe_allow_html=True
    )
    st.markdown("<h5 style='color: white !important; margin-top: 0px;'><i>Computer Vision Utility</i></h5>", unsafe_allow_html=True)

    # 1. Sidebar Upload & Controls
    st.sidebar.subheader("📁 Image Input")
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert the file to an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        st.sidebar.markdown("---")
        st.sidebar.subheader("🛠️ Processing Controls")
        
        option = st.sidebar.selectbox(
            'Select Operation',
            ('Brightness & Contrast', 'Global Thresholding', 'Adaptive Thresholding')
        )

        # Tabs for better layout
        tab1, tab2 = st.tabs(["📸 Preview & Process", "📊 Image Metrics"])

        with tab1:
            processed_img = None

            # --- OPERATION 1: BRIGHTNESS & CONTRAST ---
            if option == 'Brightness & Contrast':
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    brightness = st.slider("Brightness", -100, 100, 0, key="bright")
                with col_c2:
                    contrast = st.slider("Contrast", 0.0, 3.0, 1.0, key="contrast")
                processed_img = cv2.convertScaleAbs(img_rgb, alpha=contrast, beta=brightness)

            # --- OPERATION 2: GLOBAL THRESHOLDING ---
            elif option == 'Global Thresholding':
                thresh_val = st.slider("Binary Threshold Value", 0, 255, 127, key="thresh")
                _, processed_img = cv2.threshold(img_gray, thresh_val, 255, cv2.THRESH_BINARY)

            # --- OPERATION 3: ADAPTIVE THRESHOLDING ---
            elif option == 'Adaptive Thresholding':
                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    method = st.radio("Method", ("Mean", "Gaussian"), horizontal=True)
                with col_a2:
                    block_size = st.slider("Block Size (Odd)", 3, 99, 11, step=2)
                c_val = st.slider("Constant C", 0, 20, 2)

                adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C if method == "Mean" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
                processed_img = cv2.adaptiveThreshold(
                    img_gray, 255, adaptive_method, cv2.THRESH_BINARY, block_size, c_val
                )

            # Display Results
            st.markdown("---")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown("**Original Image**")
                st.image(img_rgb, width="stretch")
            
            with res_col2:
                st.markdown("**Processed Output**")
                if processed_img is not None:
                    st.image(processed_img, width="stretch")

        with tab2:
            st.header("Image Metadata")
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.metric("Width", f"{img.shape[1]} px")
                st.metric("Height", f"{img.shape[0]} px")
            with info_col2:
                st.metric("Channels", f"{img.shape[2] if len(img.shape) > 2 else 1}")
                st.metric("Dtype", str(img.dtype))
            
            st.info("💡 **Tip**: Use Adaptive Thresholding for images with uneven lighting.")

    else:
        st.container()
        col_m1, col_m2, col_m3 = st.columns([1, 2, 1])
        with col_m2:
            st.markdown("""
                <div style='text-align: center; padding: 50px; border: 2px dashed rgba(255,255,255,0.1); border-radius: 20px;'>
                    <h2 style='color: #007ACC !important; font-weight: 800;'>Image Processing Studio</h2>
                    <p style='color: #007ACC !important; font-weight: 600;'>Upload an image in the sidebar to begin processing.</p>
                </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: white;'>Ana-Ofelia Meneses (Built for CAI2840C Computer Vision with Streamlit)</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
