import streamlit as st
import cv2 as cv
import numpy as np
import easyocr
from streamlit_cropper import st_cropper
from PIL import Image
from gtts import gTTS
from skimage.filters import unsharp_mask


st.set_page_config(
    page_title="Image to Text OCR",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_reader(language):
    return easyocr.Reader([language], gpu=False)

def ocr(src_img):
    if src_img is not None:
        bytes_data = src_img.getvalue()
        np_img = np.frombuffer(bytes_data, np.uint8)
        img = cv.imdecode(np_img, cv.IMREAD_COLOR)
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # CROPPING & CONTROLS SIDEBAR
        with st.sidebar:
            st.subheader("Cropping tool")
            realtime_update = st.checkbox(label="Update in Real Time", value=True)
            box_color = st.color_picker(label="Box Color", value='#0000FF')
            aspect_choice = st.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])

        aspect_dict = {
            "1:1": (1, 1),
            "16:9": (16, 9),
            "4:3": (4, 3),
            "2:3": (2, 3),
            "Free": None
        }
        aspect_ratio = aspect_dict[aspect_choice]
        
        if rgb_img is not None:
            opened_rgb = Image.fromarray(rgb_img)
            if not realtime_update:
                st.write("Double click to save crop")
            
            cropped_img = st_cropper(opened_rgb, realtime_update=realtime_update, box_color=box_color, aspect_ratio=aspect_ratio)
            array_cropped_img = np.array(cropped_img)
            gray_img=cv.cvtColor(array_cropped_img, cv.COLOR_RGB2GRAY)
            sharpened = unsharp_mask(gray_img, radius=5, amount=2)
            ocr_input=(np.clip(sharpened, 0, 1) * 255).astype(np.uint8)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Cropped Region")
                st.image(cropped_img, use_container_width=True)

            with col2:
                st.subheader("Optimized OCR Input")
                st.image(ocr_input, use_container_width=True)
                
            # LANGUAGE & EXTRACTION
            lang_dict = {"English": "en", "Korean": "ko", "Japanese": "ja", "Hindi": "hi"}
            # 1. Get the user's choice. If empty, option is None.
            option = st.selectbox(
                "Language",
                options=list(lang_dict.keys()),
                placeholder="Select a language",
                index=None  # Use None so the placeholder actually shows up!
            )

            # 2. Look up the language code, or use "en" as the absolute backup.
            selected_value = lang_dict.get(option, "en")
            reader = load_reader(selected_value)
            
            result = reader.readtext(ocr_input, detail=0, paragraph=True)

            if result:
                st.write(result)
                full_text = "\n".join(result)
                try:
                    with st.spinner("Generating audio..."):
                        speech = gTTS(text=full_text, lang=selected_value, slow=False)
                        audio = "ocr.mp3"
                        speech.save(audio)
                    st.audio(audio, format="audio/mp3")
                    with open(audio, "rb") as file:
                        st.download_button(
                            label="Download Audio (MP3)",
                            data=file,
                            file_name="ocr_speech.mp3",
                            mime="audio/mp3"
                        )
                except Exception as e:
                    st.error(f"Could not generate audio. Please check your internet connection.\n {e}")
                st.download_button("Download Text", full_text, file_name="ocr.txt")
            else:
                st.warning("No text detected. Try adjusting the crop or threshold parameters.")
    else:
        st.info("Please upload before you proceed!")

with st.popover("Instructions"):
    st.markdown("""
    ### How to Use This OCR Tool 🔍
    **Step 1 – Provide an Image** -> Choose Upload a File or Use Camera.
    **Step 2 – Crop the Region** -> Adjust Aspect Ratio in sidebar.
    **Step 3 – Tune Pre-processing** -> Look at the **Blur Analysis Map**. Move the threshold slider to shift whether an image gets sharpened or smoothed dynamically.
    """)
    
input_method = st.radio("Choose how to provide the image:", ("Upload a File", "Use Camera"))
if input_method == "Upload a File":
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])
    ocr(uploaded_file)
elif input_method == "Use Camera":
    enable = st.checkbox("Enable camera")
    picture = st.camera_input("Take a picture", disabled=not enable)
    ocr(picture)