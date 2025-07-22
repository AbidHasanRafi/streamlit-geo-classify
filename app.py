import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# Configuration (should match your training config)
class Config:
    NUM_CLASSES = 7
    IMAGE_SIZE = (224, 224)
    GROUP_NAMES = [
        'transport', 'urban', 'sports', 'water', 
        'industrial', 'vegetation', 'other'
    ]

# Define the model architecture (must match training)
class SimplifiedSTModel(nn.Module):
    def __init__(self, num_classes=Config.NUM_CLASSES):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = x.squeeze(1)  # Remove temporal dimension if present
        features = self.encoder(x)
        return self.classifier(features.view(features.size(0), -1))

# Load the trained model
@st.cache_resource
def load_model(model_path):
    model = SimplifiedSTModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Make prediction
def predict(image, model):
    preprocessed = preprocess_image(image)
    with torch.no_grad():
        outputs = model(preprocessed)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        _, predicted_class = torch.max(outputs, 1)
    return probabilities, predicted_class

# Streamlit app
def main():
    st.set_page_config(page_title="PatternNet Classifier", layout="wide")
    
    st.title("PatternNet Image Classification")
    st.write("Upload an image to classify it into one of the 7 land use categories")
    
    # Sidebar
    st.sidebar.markdown("### Want to try with images?")
    st.sidebar.write("Download sample images from the PatternNet dataset:")
    st.sidebar.markdown(
    "[![Kaggle](https://img.shields.io/badge/Kaggle-PatternNet_Dataset-blue?logo=kaggle)](https://www.kaggle.com/datasets/abidhasanrafi/patternnet)"
    )
    st.sidebar.title("Options")
    model_path = st.sidebar.text_input("Model path", "best_model.pth")
    show_probabilities = st.sidebar.checkbox("Show detailed probabilities", True)
    
    # Main content
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Load model and make prediction
        try:
            model = load_model(model_path)
            
            # Add a predict button
            if st.button("Classify Image"):
                with st.spinner("Classifying..."):
                    probabilities, predicted_class = predict(image, model)
                
                # Display results
                st.success(f"Predicted Class: **{Config.GROUP_NAMES[predicted_class]}**")
                
                if show_probabilities:
                    st.subheader("Class Probabilities:")
                    for i, prob in enumerate(probabilities):
                        # Highlight the predicted class
                        if i == predicted_class:
                            st.markdown(f"**{Config.GROUP_NAMES[i]}**: {prob:.4f} ✅")
                        else:
                            st.write(f"{Config.GROUP_NAMES[i]}: {prob:.4f}")
                    
                    # Visualize probabilities as a bar chart
                    st.subheader("Probability Distribution")
                    prob_dict = {Config.GROUP_NAMES[i]: float(prob) for i, prob in enumerate(probabilities)}
                    st.bar_chart(prob_dict)
        
        except Exception as e:
            st.error(f"Error loading model or making prediction: {str(e)}")
            st.error("Please ensure:")
            st.error("1. The model file 'best_model.pth' exists in the correct location")
            st.error("2. The model architecture matches the training code")
    
    # Add some information about the classes
    st.sidebar.markdown("### Class Categories")
    for i, name in enumerate(Config.GROUP_NAMES):
        st.sidebar.write(f"{i+1}. {name.capitalize()}")

if __name__ == "__main__":
    main()