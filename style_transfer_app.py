import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import concurrent.futures
from PIL import Image
import requests
from io import BytesIO
import time

# Adjustment to TensorFlow GPU memory alloaction (not all at once)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

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
    #plt.grid(False)
    #plt.xticks([])
    #plt.yticks([])
    #plt.imshow(img)
    #st.pyplot(plt)
    st.image(img, use_column_width=True)
    
    
# Define content and style models
content_layer = 'block5_conv2'
style_layers = ['block1_conv1', 'block3_conv1', 'block5_conv1']

content_model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer(content_layer).output)
style_models = [Model(inputs=vgg_model.input, outputs=vgg_model.get_layer(layer).output) for layer in style_layers]

# Cost functions
def content_cost(content, generated):
    a_C = content_model(content)
    a_G = content_model(generated)
    return tf.reduce_mean(tf.square(a_C - a_G))

def gram_matrix(A):
    n_C = int(A.shape[-1])
    a = tf.reshape(A, [-1, n_C])
    n = tf.shape(a)[0]
    G = tf.matmul(a, a, transpose_a=True)
    return G / tf.cast(n, tf.float32)

def style_cost(style, generated):
    J_style = 0
    lam = 1. / len(style_models)
    for style_model in style_models:
        a_S = style_model(style)
        a_G = style_model(generated)
        GS = gram_matrix(a_S)
        GG = gram_matrix(a_G)
        current_cost = tf.reduce_mean(tf.square(GS - GG))
        J_style += current_cost * lam
    return J_style

# Training loop
def training_loop(content, style, iterations=20, alpha=10., beta=20.):
    generated = tf.Variable(content, dtype=tf.float32)
    opt = Adam(learning_rate=7.)
    best_cost = float('inf')
    best_image = generated.numpy()
    generated_images = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()
    
    batch_size = 10
    num_batches = content.shape[0] // batch_size # hpw many batches are needed to process all imgs (content and style arrays)
    
    for i in range(iterations):
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = (batch_idx + 1) * batch_size
            content_batch = content[start_idx:end_idx]
            style_batch = style[start_idx:end_idx]
        
            with tf.GradientTape() as tape:
                J_content = content_cost(content_batch, generated)
                J_style = style_cost(style_batch, generated)
                J_total = alpha * J_content + beta * J_style
            
            grads = tape.gradient(J_total, generated)
            opt.apply_gradients([(grads, generated)])
            
            # Update best_image if current J_total is better
            if J_total < best_cost:
                best_cost = J_total
                best_image = generated.numpy()
                
            elapsed_time = time.time() - start_time
            status_text.text(f"Iteration {i+1}/{iterations}, Cost: {J_total:.2f}, Time elapsed: {elapsed_time:.2f}s")
            progress_bar.progress((i+1) / iterations)
        generated_images.append(generated.numpy())
    return best_image



# Streamlit interface
cols = st.columns(2)

content_img = None
style_img = None

use_local_content = cols[0].checkbox("Use Local Content Image", value=False)
use_local_style = cols[1].checkbox("Use Local Style Image", value=False)

# Content Image Inputs
if use_local_content:
    content_file = cols[0].file_uploader("Choose Content Image...", type=["jpg", "png", "jpeg"], key='content')
    if content_file:
        content_img = Image.open(content_file)
        content_array = load_and_process_image(content_file)
else:
    content_url = cols[0].text_input("Enter Content Image URL", "")
    if content_url:
        content_img = Image.open(BytesIO(requests.get(content_url).content))
        content_array = load_and_process_image_url(content_url)

# Style Image Inputs        
if use_local_style:
    style_file = cols[1].file_uploader("Choose Style Image...", type=["jpg", "png", "jpeg"], key='style')
    if style_file:
        style_img = Image.open(style_file)
        style_array = load_and_process_image(style_file)
else:
    style_url = cols[1].text_input("Enter Style Image URL", "")
    if style_url:
        style_img = Image.open(BytesIO(requests.get(style_url).content))
        style_array = load_and_process_image_url(style_url)

iterations = st.slider("Number of iterations", min_value=10, max_value=100, value=20, step=10)

if content_img and style_img:
    cols[0].image(content_img, caption='Content Image', use_column_width=True)
    cols[1].image(style_img, caption='Style Image', use_column_width=True)

    if st.button("Generate"):
        with st.spinner('Processing...'):
            best_image = training_loop(content_array, style_array, iterations=iterations)
            best_image = deprocess(best_image)  # Deprocess the image
            st.image(best_image, caption='Stylized Image')


