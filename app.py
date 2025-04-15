import streamlit as st
import cv2
import numpy as np
import pywt
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

# Load Pretrained CNN Model (VGG16)
@st.cache_resource
def get_feature_extractor():
    base_model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer("block3_conv3").output)
    return model

# Extract Features using CNN
def extract_features(image, model):
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=-1)
    image = np.repeat(image, 3, axis=-1)
    image = np.expand_dims(image, axis=0)
    return model.predict(image)

# DWT-Based Fusion with CNN Features
def dwt_cnn_fusion(image1, image2):
    image1 = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2GRAY)
    image2 = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2GRAY)

    image1 = cv2.resize(image1, (min(image1.shape[1], image2.shape[1]), min(image1.shape[0], image2.shape[0])))
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    coeffs1 = pywt.dwt2(image1, 'haar')
    coeffs2 = pywt.dwt2(image2, 'haar')

    cA1, (cH1, cV1, cD1) = coeffs1
    cA2, (cH2, cV2, cD2) = coeffs2

    # Extract CNN Features
    cnn_model = get_feature_extractor()
    cH1_features = extract_features(cH1, cnn_model)
    cH2_features = extract_features(cH2, cnn_model)

    # Reshape CNN features
    cH1_features = cv2.resize(cH1_features[0, :, :, 0], (cH1.shape[1], cH1.shape[0]))  
    cH2_features = cv2.resize(cH2_features[0, :, :, 0], (cH2.shape[1], cH2.shape[0]))

    # Fusion Strategy
    cA = (cA1 + cA2) / 2
    cH = np.maximum(cH1_features, cH2_features)
    cV = np.maximum(cV1, cV2)
    cD = np.maximum(cD1, cD2)

    # Apply Inverse DWT
    fused_image = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
    fused_image = np.uint8(np.clip(fused_image, 0, 255))
    return fused_image

# Image Segmentation
def segment_image(image):
    _, segmented = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    return segmented

# Histogram Plot
def plot_histogram(image):
    fig, ax = plt.subplots()
    ax.hist(image.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.7)
    ax.set_title("Histogram of Fused Image")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# Streamlit Web UI
st.title("Medical Image Fusion: Wavelet-CNN Approach")

# File Uploaders
uploaded_file1 = st.file_uploader("Upload CT Image", type=["jpg", "png", "jpeg"])
uploaded_file2 = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file1 and uploaded_file2:
    image1 = Image.open(uploaded_file1)
    image2 = Image.open(uploaded_file2)

    st.image([image1, image2], caption=["CT Image", "MRI Image"], width=300)
    
    if st.button("Perform Fusion"):
        fused_image = dwt_cnn_fusion(image1, image2)
        segmented_image = segment_image(fused_image)

        st.image(fused_image, caption="Fused Image (Wavelet-CNN)", use_column_width=True, channels="GRAY")
        st.image(segmented_image, caption="Segmented Image", use_column_width=True, channels="GRAY")

        plot_histogram(fused_image)

        # Save Fused Image
        fused_image_pil = Image.fromarray(fused_image)
        fused_image_pil.save("fused_image.png")
        st.success("Fused Image saved as 'fused_image.png'")

        # Provide Download Option
        with open("fused_image.png", "rb") as file:
            st.download_button("Download Fused Image", file, "fused_image.png")

