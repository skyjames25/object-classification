# %%
import streamlit as st
from PIL import Image
import torch
from torchvision.transforms import transforms
import os
from scene_train import ConvNet, transformer,classes


# Check for GPU availability and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%
model = ConvNet(num_classes=6).to(device)
model.load_state_dict(torch.load(r'C:\Users\skyja\Desktop\Projects\visualcode\scene_detector\best_checkpoint.model', map_location=torch.device('cpu')))
model.eval()

def prediction(image, transformer, model, classes):
    image = transformer(image).float()
    image = image.unsqueeze_(0).to(device)  # Ensure the image is on the same device as the model
    output = model(image)
    index = output.data.cpu().numpy().argmax()  # Move output to CPU before converting to numpy
    pred = classes[index]
    return pred

# %%
# Define the transformer
# Streamlit UI
st.title("Image Classification with PyTorch")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = prediction(image, transformer, model, classes)
    st.write(f"Prediction: {label}")



# %%



