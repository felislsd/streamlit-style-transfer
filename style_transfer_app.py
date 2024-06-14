import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import time

# Streamlit UI setup
st.set_page_config(layout="wide")
st.title("Neural Style Transfer")

# Load VGG19 model
@st.cache_resource
def load_vgg19_model():
    model = VGG19(include_top=False, weights='imagenet')
    model.trainable = False
    return model

vgg_model = load_vgg19_model()


# Image processing functions
def load_and_process_image(image_path):
    img = load_img(image_path)
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0) #model expects 4dim tensor
    return img

def load_and_process_image_url(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

def deprocess(x):
    x[:, :, 0] += 103.939 # does the opposite of preprocess_input
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]  # invert the order of the channels
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def display_image(image):
    img = np.squeeze(image, axis=0)
    img = deprocess(img)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    st.pyplot(plt)
