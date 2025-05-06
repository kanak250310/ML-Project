import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import gdown
import os

# Download the model file if not present
if not os.path.exists('STL_Safe.pt'):
    url = 'https://drive.google.com/uc?id=1yw4bfjy3X5fOVk37VI9VYFhyQaqTeIVA'
    output = 'STL_Safe.pt'
    with st.spinner('Downloading model... (this may take a while)'):
        gdown.download(url, output, quiet=False)

# Load the model
@st.cache_resource
def load_model():
    if not os.path.exists('STL_Safe.pt'):
        st.error("Model file not found! Please ensure STL_Safe.pt is in the directory.")
        st.stop()
    
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 10)
    model.load_state_dict(torch.load('STL_Safe.pt', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Class names for STL10
class_names = ['airplane', 'bird', 'car', 'cat', 'deer', 
               'dog', 'horse', 'monkey', 'ship', 'truck']

# Image transformation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Streamlit app
st.title("SafeStudent OSSL Model Demo")
st.write("Classify images using the trained SafeStudent model on STL10 dataset")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess and predict
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
    
    # Display prediction
    st.subheader("Prediction Results")
    st.write(f"**Predicted class:** {class_names[predicted.item()]}")
    
    # Show confidence scores
    st.write("**Confidence scores:**")
    for i, (name, prob) in enumerate(zip(class_names, probabilities)):
        st.write(f"{name}: {prob:.1f}%")
    
    # Visualize the confidence scores
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(class_names, probabilities.numpy())
    ax.set_xlabel('Confidence (%)')
    ax.set_title('Classification Confidence')
    ax.set_xlim(0, 100)
    st.pyplot(fig)

st.sidebar.markdown("""
### About this app
This app uses a SafeStudent OSSL model trained on the STL10 dataset.

**Model details:**
- Architecture: ResNet18
- Training: Semi-supervised learning
- Classes: 10 STL10 categories

Upload any image to see how the model classifies it!
""")
