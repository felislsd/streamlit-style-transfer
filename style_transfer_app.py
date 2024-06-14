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
