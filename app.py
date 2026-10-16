import os
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageFilter

MODEL_PATH = "models/digit_cnn.keras"

st.set_page_config(page_title="Digit Recognition AI", page_icon="🔢", layout="centered")
st.title("🔢 Handwritten Digit Recognition")
st.write("A beginner-friendly Deep Learning project using a Convolutional Neural Network (CNN).")
st.write("Upload an image containing one handwritten digit (0–9), and the model will predict it.")

@st.cache_resource
def load_model():
    try:
        from tensorflow.keras.models import load_model as keras_load_model
        if os.path.exists(MODEL_PATH):
            return keras_load_model(MODEL_PATH)
    except Exception:
        return None
    return None

def preprocess_image(image):
    """Convert an uploaded image into a 28x28 MNIST-style input."""
    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)
    # MNIST uses a dark background and a light digit. Invert typical paper images.
    pixels = np.asarray(image)
    if pixels.mean() > 127:
        image = ImageOps.invert(image)
    image = image.filter(ImageFilter.GaussianBlur(radius=0.4))
    image.thumbnail((20, 20))
    canvas = Image.new("L", (28, 28), color=0)
    x = (28 - image.width) // 2
    y = (28 - image.height) // 2
    canvas.paste(image, (x, y))
    arr = np.asarray(canvas).astype("float32") / 255.0
    return arr.reshape(1, 28, 28, 1), canvas

uploaded = st.file_uploader("Choose a digit image", type=["png", "jpg", "jpeg", "bmp"])

if uploaded:
    original = Image.open(uploaded).convert("RGB")
    st.image(original, caption="Uploaded image", width=240)
    model = load_model()
    if model is None:
        st.warning("The trained model was not found. Run `python train.py` first to create it.")
        st.code("python train.py\nstreamlit run app.py", language="bash")
    else:
        input_array, preview = preprocess_image(original)
        with st.expander("See the processed 28 × 28 image"):
            st.image(preview.resize((280, 280)), caption="Model input", width=280)
        probabilities = model.predict(input_array, verbose=0)[0]
        prediction = int(np.argmax(probabilities))
        st.success(f"Predicted digit: **{prediction}**")
        st.metric("Model confidence", f"{probabilities[prediction] * 100:.1f}%")
        st.subheader("Probability for each digit")
        st.bar_chart({str(i): float(probabilities[i]) for i in range(10)})
else:
    st.info("Tip: use a clear image with one large digit, preferably dark ink on a plain light background.")
    st.markdown("### How it works")
    st.markdown("1. A CNN learns patterns from the MNIST handwritten-digit dataset.\n"
                "2. The uploaded image is converted to grayscale and resized to 28 × 28 pixels.\n"
                "3. The model outputs a probability for each digit from 0 to 9.")
