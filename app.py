import streamlit as st
from PIL import Image
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor

# Set the title of the Streamlit app
st.title("🔬 Skin Cancer Classification App")
st.write("Upload an image of a skin lesion, and the model will predict its type.")

# --- Model Loading ---
# Using @st.cache_resource to load the model only once
@st.cache_resource
def load_model():
    """Loads the pre-trained model and the correct feature extractor."""
    model_path = "./skin-cancer-classification"
    
    # AutoFeatureExtractor will correctly load the Swin preprocessor
    # based on the model_type in config.json
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
    model = AutoModelForImageClassification.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model, feature_extractor, device

# Load the model, feature extractor, and device
model, feature_extractor, device = load_model()

# --- Image Uploader and Prediction ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    inputs = feature_extractor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_class_idx]

    st.success(f"**Prediction:** {predicted_label}")
else:
    st.info("Please upload an image file to get a prediction.")

st.markdown("---")
st.write("App built with Streamlit and Hugging Face Transformers.")