import streamlit as st
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# Define the VAE model (same as in training script)
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # Mean
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # Log variance
        # Decoder
        self.fc3 = nn.Linear(latent_dim + 10, hidden_dim)  # Conditioned on digit
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c):
        h = self.relu(self.fc1(x))
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        h = self.relu(self.fc3(torch.cat([z, c], dim=1)))
        return self.sigmoid(self.fc4(h))

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, 784), c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

# Load the trained model
@st.cache_resource
def load_model():
    model = VAE()
    model_path = "vae_mnist.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        st.error("Model weights not found. Please ensure 'vae_mnist.pth' is in the same directory.")
    model.eval()
    return model

# Generate images for a selected digit
def generate_images(model, digit, num_images=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Create one-hot encoding for the digit
    c = torch.zeros(1, 10).to(device)
    c[0, digit] = 1
    # Generate images
    images = []
    with torch.no_grad():
        for _ in range(num_images):
            z = torch.randn(1, 20).to(device)  # Sample from latent space
            generated = model.decode(z, c).cpu().numpy().reshape(28, 28)
            images.append(generated)
    return images

# Streamlit app
st.title("Handwritten Digit Generator")
st.write("Select a digit (0-9) to generate 5 handwritten digit images using a Variational Autoencoder.")

# User input
digit = st.selectbox("Select a digit:", list(range(10)))

# Generate button
if st.button("Generate Images"):
    model = load_model()
    images = generate_images(model, digit)
    
    # Display images in a grid
    st.write(f"Generated images for digit {digit}:")
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i, img in enumerate(images):
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    st.pyplot(fig)

st.write("Note: This app uses a trained VAE model to generate MNIST-like images. Ensure the model weights ('vae_mnist.pth') are available.")