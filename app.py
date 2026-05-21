import streamlit as st
import cv2 as cv
import numpy as np
import easyocr
from streamlit_cropper import st_cropper
from PIL import Image
from gtts import gTTS
from io import BytesIO

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Image to Text OCR",
    page_icon="🔠",
    layout="wide"
)

# --- CACHED OCR ENGINE ---
@st.cache_resource
def load_reader(language_code):
    return easyocr.Reader([language_code], gpu=False)

# --- PERMANENT CONTROL SIDEBAR ---
with st.sidebar:
    st.header("🛠️ Image Configurations")
    
    st.subheader("Cropping Parameters")
    realtime_update = st.checkbox(label="Update in Real Time", value=True)
    box_color = st.color_picker(label="Crop Frame Color", value='#0000FF')
    aspect_choice = st.radio(
        label="Target Aspect Ratio", 
        options=["Free", "1:1", "16:9", "4:3", "2:3"]
    )
    
    st.divider()
    st.subheader("Image Pre-processing")
    binary_threshold = st.slider(
        label="Binarization Manual Threshold", 
        min_value=30, 
        max_value=220, 
        value=127,
        help="Adjust if text lines look broken or too thick."
    )

# Map Aspect Ratio selections cleanly
aspect_dict = {"1:1": (1, 1), "16:9": (16, 9), "4:3": (4, 3), "2:3": (2, 3), "Free": None}
aspect_ratio = aspect_dict[aspect_choice]

# --- MAIN PAGE HEADER ---
st.title("🔍 Advanced Image to Text OCR Converter")
st.write("Extract typography instantly and generate multi-language voice outputs.")

with st.popover("📖 Instructions"):
    st.markdown("""
    ### How to Use This OCR Tool
    1. **Provide an Image:** Select either file upload or system camera capture below.
    2. **Refine Target Selection:** Move or resize the bounding frame to target specific paragraphs. 
    3. **Set Processing Adjustments:** If text is light or low-contrast, adjust the **Binarization Manual Threshold** in the sidebar to make character lines cleaner.
    """)

# --- MEDIA SOURCE ACQUISITION ---
input_method = st.radio("Select Image Input Source:", ("Upload a File", "Use Camera"), horizontal=True)
src_img = None

if input_method == "Upload a File":
    src_img = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])
else:
    enable = st.checkbox("Power on Camera hardware")
    if enable:
        src_img = st.camera_input("Capture frame")

st.divider()

# --- MAIN EXECUTION ENGINE ---
if src_img is not None:
    # Read stream bytes cleanly to an OpenCV Image matrix
    bytes_data = src_img.getvalue()
    np_img = np.frombuffer(bytes_data, np.uint8)
    img = cv.imdecode(np_img, cv.IMREAD_COLOR)
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    opened_rgb = Image.fromarray(rgb_img)

    if not realtime_update:
        st.info("💡 Tip: Double-click inside the selection box to manually trigger processing updates.")

    # Execute Image Cropping Module
    cropped_img = st_cropper(
        opened_rgb, 
        realtime_update=realtime_update, 
        box_color=box_color, 
        aspect_ratio=aspect_ratio
    )
    
    if cropped_img:
        array_cropped_img = np.array(cropped_img)
        
        # --- PRE-PROCESSING PIPELINE ---
        gray_img = cv.cvtColor(array_cropped_img, cv.COLOR_RGB2GRAY)
        
        # Optimize contrast dynamically across regions using CLAHE
        clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray_img)
        
        # Isolate text elements clearly using adaptive threshold masking
        _, threshold_mask = cv.threshold(clahe_img, binary_threshold, 255, cv.THRESH_BINARY)
        
        # OCR feeds best on cleanly defined high-contrast binarized frames
        ocr_input = threshold_mask

        # Render Source vs Processing Side-by-Side Comparison
        view_col1, view_col2 = st.columns(2)
        with view_col1:
            st.subheader("🎯 Cropped Focus Region")
            st.image(cropped_img, use_container_width=True)
        with view_col2:
            st.subheader("⚙️ Optimized Machine Input")
            st.image(ocr_input, use_container_width=True, channels="GRAY")
        
        st.divider()

        # --- LANG SELECT & MODEL INFERENCE ---
        st.subheader("📝 Text Extraction & Synthesis")
        
        lang_dict = {"English": "en", "Korean": "ko", "Japanese": "ja", "Hindi": "hi"}
        option = st.selectbox(
            "Target Document Language",
            options=list(lang_dict.keys()),
            index=0  # Defaults to English out of the gate
        )
        selected_value = lang_dict.get(option, "en")
        
        # Run inference using cached easyocr instances
        reader = load_reader(selected_value)
        with st.spinner("Analyzing structural character layers..."):
            result = reader.readtext(ocr_input, detail=0, paragraph=True)

        if result:
            full_text = "\n".join(result)
            
            # Displays extracted content cleanly in an explicit, easily editable container
            st.text_area("Extracted Characters Output", value=full_text, height=250)
            
            # Action Buttons Layout
            btn_col1, btn_col2 = st.columns(2)
            
            with btn_col1:
                st.download_button(
                    label="📄 Download Text File (.txt)", 
                    data=full_text, 
                    file_name="ocr_output.txt",
                    width="stretch"
                )
            
            # Text-To-Speech Synthesis via In-Memory Data Streams
            with btn_col2:
                try:
                    with st.spinner("Synthesizing audio stream..."):
                        audio_buffer = BytesIO()
                        speech = gTTS(text=full_text, lang=selected_value, slow=False)
                        speech.write_to_fp(audio_buffer)
                        audio_buffer.seek(0)  # Rewind pointer to beginning of file data
                    
                    st.audio(audio_buffer, format="audio/mp3")
                    st.download_button(
                        label="🔊 Download Voice Track (.mp3)",
                        data=audio_buffer,
                        file_name="ocr_speech_track.mp3",
                        mime="audio/mp3",
                        width="stretch"
                    )
                except Exception as e:
                    st.error(f"Text-to-Speech system could not run. Verify active internet access. Trace: {e}")
        else:
            st.warning("No clear character matches discovered. Try widening your frame or sliding the threshold bar.")
else:
    st.info("👋 System ready. Please feed an image source to begin data transformation extraction.")