import streamlit as st
import pandas as pd
import os
from PIL import Image

# ---------------- Style ----------- #
st.markdown(
    """
    <style>
    /* Base button styling */
    .stButton>button {
        border: none !important;
        background-color: #42ff5b !important;
        color: black !important;
        position: relative;
        overflow: visible !important;
        z-index: 0;
        transition: all 0.2s ease;
    }

    /* Glow effect */
    .stButton>button::before {
        content: "";
        background: linear-gradient(
            45deg,
            #FF0000, #FF7300, #FFFB00, #48FF00,
            #00FFD5, #002BFF, #FF00C8, #FF0000
        );
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background-size: 400%;
        z-index: -1;
        filter: blur(5px);
        animation: glowing 20s linear infinite;
        opacity: 0;
        border-radius: 8px;
        transition: opacity 0.2s ease;
    }

    /* Hover state */
    .stButton>button:hover::before {
        opacity: 0.7;
    }

    /* Active/Clicked state */
    .stButton>button:active::before {
        opacity: 1 !important;
        filter: blur(5px) brightness(1.2);
        animation-duration: 1s;
    }

    @keyframes glowing {
        0% { background-position: 0 0; }
        50% { background-position: 400% 0; }
        100% { background-position: 0 0; }
    }
    </style>
    """,
    unsafe_allow_html=True
)
# --------------- End style -------- #

st.title("Parkinson detection using YOLO v8")
st.divider()
st.subheader("Metrics Graphs")
st.divider()

file_path = os.path.join(os.path.dirname(__file__), 'results.csv')
df = pd.read_csv(file_path)
geo_df = pd.read_csv(file_path)

exclude_columns = ['time']

plot_columns = [col for col in df.columns
                if col not in exclude_columns
                and pd.api.types.is_numeric_dtype(df[col])]

for col in plot_columns:
    with st.expander(f"📈 {col}", expanded=False):
        st.line_chart(df, x='epoch', y=col)

tab1, tab2 = st.tabs(["📉 Individual Metrics", "📊 Combined View"])
with tab1:
    for col in plot_columns:
        st.subheader(col)
        st.line_chart(df, x='epoch', y=col)

with tab2:
    st.line_chart(df, x='epoch', y=plot_columns)

st.subheader("Metrics Table")
st.divider()

# DataFrame
st.dataframe(df)

st.text("")
st.subheader("Confusion matrix")
st.divider()

IMAGE_DIR = 'Graphs'
STANDARD_IMG = os.path.join(IMAGE_DIR, 'confusion_matrix.png')
NORMALIZED_IMG = os.path.join(IMAGE_DIR, 'confusion_matrix_normalized.png')

@st.cache_data(ttl=3600)
def load_image(image_path):
    """Load and cache an image file"""
    try:
        return Image.open(image_path)
    except FileNotFoundError:
        st.error(f"Image not found: {image_path}")
        return None
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

if not os.path.exists(STANDARD_IMG):
    st.error(f"Image not found: {STANDARD_IMG}")
if not os.path.exists(NORMALIZED_IMG):
    st.error(f"Image not found: {NORMALIZED_IMG}")

matrix_type = st.radio(
    "Select confusion matrix type:",
    options=["Standard", "Normalized"],
    horizontal=True,
    index=0,
    help="Standard shows raw counts, Normalized shows percentages"
)

if matrix_type == "Standard":
    img = load_image(STANDARD_IMG)
    if img:
        st.image(img,
                caption="Confusion Matrix (Raw Counts)",
                use_container_width=True)
else:
    img = load_image(NORMALIZED_IMG)
    if img:
        st.image(img,
                caption="Normalized Confusion Matrix (Percentages)",
                use_container_width=True)

if st.button("Clear Cache"):
    st.cache_data.clear()
    st.success("Image cache cleared!")

st.divider()
PREDICTIONS_DIR = 'Predictions'


@st.cache_data
def get_prediction_images():
    """Get all prediction images from folder with caching"""
    images = []
    valid_extensions = ('.png', '.jpg', '.jpeg')

    if os.path.exists(PREDICTIONS_DIR):
        for file in sorted(os.listdir(PREDICTIONS_DIR)):
            if file.lower().endswith(valid_extensions):
                try:
                    img_path = os.path.join(PREDICTIONS_DIR, file)
                    images.append((file, Image.open(img_path)))
                except Exception as e:
                    st.error(f"Error loading {file}: {str(e)}")
    else:
        st.error(f"Directory not found: {PREDICTIONS_DIR}")

    return images

st.title("Model Predictions Viewer")

prediction_images = get_prediction_images()

if prediction_images:
    st.sidebar.header("Display Options")

    selected_file = st.sidebar.selectbox(
        "Choose an image to display:",
        options=[img[0] for img in prediction_images],
        index=0
    )

    compare_images = st.sidebar.multiselect(
        "Compare multiple images:",
        options=[img[0] for img in prediction_images],
        default=[prediction_images[0][0]]
    )

    img_index = st.sidebar.slider(
        "Browse images:",
        0, len(prediction_images) - 1, 0,
        help="Use slider to quickly browse through images"
    )

    st.header("Selected Prediction")

    selected_img = next(img for name, img in prediction_images if name == selected_file)
    st.image(selected_img,
             caption=f"Selected: {selected_file}",
             use_container_width=True)

    if len(compare_images) > 1:
        st.header("Comparison View")
        cols = st.columns(len(compare_images))
        for col, img_name in zip(cols, compare_images):
            img = next(img for name, img in prediction_images if name == img_name)
            col.image(img, caption=img_name, use_container_width=True)


    st.header("Quick Browse")
    slider_img = prediction_images[img_index][1]
    st.image(slider_img,
             caption=f"Image {img_index + 1}/{len(prediction_images)}: {prediction_images[img_index][0]}",
             use_container_width=True)

else:
    st.warning("No prediction images found in the Predictions folder")