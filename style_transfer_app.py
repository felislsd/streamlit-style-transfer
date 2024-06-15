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
import logging
from tensorflow.python.client import device_lib
#import os
import os
import gc
from tensorflow.keras import backend as K


# Set TensorFlow GPU allocator
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Clear session and collect garbage to free up memory
def clear_memory():
    K.clear_session()
    gc.collect()

# List all devices (including GPU)
print(device_lib.list_local_devices())


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Clear GPU Memory
tf.keras.backend.clear_session()

# Enable Memory Fragmentation
#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

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

# Example function to monitor memory usage
def monitor_memory_usage():
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if not gpu_devices:
        print("no gpu device found")
    else:
        for gpu in gpu_devices:
            #device_name = gpu.name.split('/')[-1].replace('_', ':')
            details = tf.config.experimental.get_memory_info('GPU:0')
            print(f"Memory Details: {details}")
            

def resize_image(image, target_height, target_width):
    if original_width >= original_height:
    # Landscape orientation or square image
        new_width = target_size
        new_height = int(original_height * target_size / original_width)
    else:
    # Portrait orientation
        new_height = target_size
        new_width = int(original_width * target_size / original_height)
    resized_image = image.resize((new_width, new_height))
    return resized_image

target_size = 512

# Image processing functions
def load_and_process_image(image_path, original_width, original_height ):
    img = load_img(image_path)
    img = resize_image(img, original_width, original_height)
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0) #model expects 4dim tensor
    return img

def load_and_process_image_url(image_url, original_width, original_height ):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img = resize_image(img, original_width, original_height)
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
    #x = np.flipud(x)
    return x

def display_image(image):
    img = np.squeeze(image, axis=0)
    img = deprocess(img)
    st.image(img, use_column_width=True)
    
    
# Define content and style models
content_layer = 'block5_conv2'
#style_layers = ['block1_conv1', 'block3_conv1', 'block5_conv1']
#style_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv2']
style_layers = ['block1_conv1', 'block2_conv2']

content_model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer(content_layer).output)
style_models = [Model(inputs=vgg_model.input, outputs=vgg_model.get_layer(layer).output) for layer in style_layers]

# Cost functions
def content_cost(content, generated):
    a_C = content_model(content)
    a_G = content_model(generated)
    return tf.reduce_mean(tf.square(a_C - a_G))

@tf.custom_gradient
def custom_matmul(a, b):
    # Implement your custom MatMul operation with gradient checkpointing
    assert a.shape[-1] == b.shape[0], f"Matmul shape mismatch: {a.shape} and {b.shape}"
    c = tf.matmul(a, b)
    
    def grad(dy):
        # Implement gradient computation
        da = tf.matmul(dy, tf.transpose(b))
        db = tf.matmul(tf.transpose(a), dy)
        return da, db

    return c, grad

def gram_matrix(A):
    n_C = int(A.shape[-1])
    a = tf.reshape(A, [-1, n_C])
    n = tf.shape(a)[0]
    G = tf.matmul(a, a, transpose_a=True)
    #G = custom_matmul(a, tf.transpose(a))
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
    logging.info(f"Starting training loop with {iterations} iterations.")
    logging.info(f"Content shape: {content.shape}")
    logging.info(f"Style shape: {style.shape}")
    generated = tf.Variable(content, dtype=tf.float32)
    opt = Adam(learning_rate=7.)
    best_cost = float('inf')
    best_image = generated.numpy()
    generated_images = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()
    
    
    for i in range(iterations):
        monitor_memory_usage()
        clear_memory()
        with tf.GradientTape() as tape:
            J_content = content_cost(content, generated)
            J_style = style_cost(style, generated)
            logging.info(f"J_content: {J_content:.2f}")
            logging.info(f"J_style: {J_style:.2f}")
            J_total = alpha * J_content + beta * J_style
            logging.info(f"J_total: {J_total:.2f}")
            
            grads = tape.gradient(J_total, generated)
            opt.apply_gradients([(grads, generated)])
            
            # Update best_image if current J_total is better
            if J_total < best_cost:
                best_cost = J_total
                best_image = generated.numpy()
                
            elapsed_time = time.time() - start_time
            #status_text.text(f"Iteration {i+1}/{iterations}, Cost: {J_total:.2f}, Time elapsed: {elapsed_time:.2f}s")
            #Print(f"Iteration {i+1}/{iterations}, Cost: {J_total:.2f}, Time elapsed: {elapsed_time:.2f}s")
            progress_bar.progress((i+1) / iterations)
            monitor_memory_usage()
        generated_images.append(generated.numpy())
        logging.info(f"Iteration {i+1}/{iterations} completed. Best cost: {best_cost:.2f}")
    logging.info(f"Training completed. Best cost: {best_cost:.2f}")
    return best_image



# Streamlit interface
cols = st.columns(2)

content_img = None
style_img = None
original_width = None
original_height = None

use_local_content = cols[0].checkbox("Use Local Content Image", value=False)
use_local_style = cols[1].checkbox("Use Local Style Image", value=False)

# Content Image Inputs
if use_local_content:
    content_file = cols[0].file_uploader("Choose Content Image...", type=["jpg", "png", "jpeg"], key='content')
    if content_file:
        content_img = Image.open(content_file)
        original_width, original_height = content_img.size
        content_array = load_and_process_image(content_file, original_width, original_height)
else:
    content_url = cols[0].text_input("Enter Content Image URL", "")
    if content_url:
        content_img = Image.open(BytesIO(requests.get(content_url).content))
        original_width, original_height = content_img.size
        content_array = load_and_process_image_url(content_url, original_width, original_height)


# Style Image Inputs        
if use_local_style:
    style_file = cols[1].file_uploader("Choose Style Image...", type=["jpg", "png", "jpeg"], key='style')
    if style_file:
        style_img = Image.open(style_file)
        original_width, original_height = style_img.size
        style_array = load_and_process_image(style_file, original_width, original_height)
else:
    style_url = cols[1].text_input("Enter Style Image URL", "")
    if style_url:
        style_img = Image.open(BytesIO(requests.get(style_url).content))
        original_width, original_height = style_img.size
        style_array = load_and_process_image_url(style_url, original_width, original_height)

iterations = st.slider("Number of iterations", min_value=10, max_value=100, value=20, step=10)

if content_img and style_img:
    cols[0].image(content_img, caption='Content Image', use_column_width=True)
    cols[1].image(style_img, caption='Style Image', use_column_width=True)

    if st.button("Generate"):
        with st.spinner('Processing...'):
            best_image = training_loop(content_array, style_array, iterations=iterations)
            display_image(best_image)
            #best_image = deprocess(best_image)  # Deprocess the image
            #st.image(best_image, caption='Stylized Image')


