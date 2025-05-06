import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import gdown  # For downloading from Google Drive
import os

# Define the model architecture (must match training)
class DeFixMatchImageModel(nn.Module):
    def __init__(self, num_classes=10):
        super(DeFixMatchImageModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 24 * 24, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Download model from Google Drive
@st.cache_resource
def load_model():
    model_url = "https://drive.google.com/uc?id=1Ujs-Z-8vMtrIxomOdGEZXNdsR9pXFRBG"
    model_path = "STL_DeFix.pt"
    
    if not os.path.exists(model_path):
        with st.spinner('Downloading model... This may take a while...'):
            gdown.download(model_url, model_path, quiet=False)
    
    model = DeFixMatchImageModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocess the image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# Class names for STL10
classes = ['airplane', 'bird', 'car', 'cat', 'deer', 
           'dog', 'horse', 'monkey', 'ship', 'truck']

# Streamlit app
st.title("STL10 Image Classifier")
st.write("Upload an image to classify it into one of 10 STL10 categories")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Load model
try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

if uploaded_file is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess and predict
        input_tensor = preprocess_image(image)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidences = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
        
        # Display top prediction
        top_pred = max(confidences.items(), key=lambda x: x[1])
        st.subheader(f"Prediction: {top_pred[0]} (confidence: {top_pred[1]:.2f})")
        
        # Show all class confidences
        st.subheader("All class confidences:")
        for class_name, confidence in sorted(confidences.items(), key=lambda x: -x[1]):
            st.write(f"{class_name}: {confidence:.4f}")
            
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
