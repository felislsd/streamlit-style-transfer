import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import streamlit as st
import concurrent.futures
from PIL import Image
import requests
from io import BytesIO
import time
import logging
from tensorflow.python.client import device_lib
import os
import gc
from tensorflow.keras import backend as K



# Set TensorFlow GPU allocator for memory fragmentation
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

def clear_memory():
    K.clear_session()
    gc.collect()


# Adjustment to TensorFlow GPU memory alloaction (not all at once)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Streamlit UI setup
st.set_page_config(layout="wide")
st.title("Neural Style Transfer")


@st.cache_resource
def load_vgg19_model():
    model = VGG19(include_top=False, weights='imagenet')
    model.trainable = False
    return model

vgg_model = load_vgg19_model()


# Image processing functions          

def resize_image(image):
    original_width, original_height = image.size
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

target_size = int(os.environ.get('TARGET_SIZE', 1024))

        

def load_and_process_img(img):
    img = img.convert('RGB') #removes alpha ch
    img = resize_image(img)
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0) #model expects 4dim tensor
    return img



def deprocess(x):
    x[:, :, 0] += 103.939 
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]  
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def display_image(image):
    img = np.squeeze(image, axis=0)
    img = deprocess(img)
    st.image(img, use_column_width=True)
    
    
# Define content and style models

# Content layer
content_layer = 'block5_conv2'


# Style layers
style_layers_list_1 = ['block1_conv1', 'block2_conv1']
style_layers_list_2 = ['block1_conv1', 'block2_conv2']
style_layers_list_3 = ['block1_conv1', 'block2_conv1', 'block3_conv1']
style_layers_list_4 = ['block1_conv1', 'block2_conv2', 'block3_conv1']
style_layers_list_5 = ['block1_conv1', 'block2_conv2', 'block3_conv3']
style_layers_list_6 = ['block1_conv1', 'block2_conv1', 'block4_conv1']
style_layers_list_7 = ['block1_conv1', 'block3_conv1', 'block4_conv1']
style_layers_list_8 = ['block1_conv1', 'block3_conv2', 'block4_conv2']
style_layers_list_9 = ['block1_conv1', 'block3_conv1', 'block5_conv1']
style_layers_list_10 = ['block1_conv1', 'block3_conv2', 'block5_conv3']
                      

style_layers_lists = {
    'Style 1': style_layers_list_1,
    'Style 2': style_layers_list_2,
    'Style 3': style_layers_list_3,
    'Style 4': style_layers_list_4,
    'Style 5': style_layers_list_5,
    'Style 6': style_layers_list_6,
    'Style 7': style_layers_list_7,
    'Style 8': style_layers_list_8,
    'Style 9': style_layers_list_9,
    'Style 10': style_layers_list_10
}


# Cost functions

def content_cost(content, generated):
    a_C = content_model(content)
    a_G = content_model(generated)
    return tf.reduce_mean(tf.square(a_C - a_G))


def gram_matrix(A):
    # The number of channels (filters) in tensor A
    n_C = int(A.shape[-1])
    # Reshapes tensor A to combine all spatial dimensions into a single dimension
    a = tf.reshape(A, [-1, n_C])
    # Computes the number of elements in the reshaped tensor A
    n = tf.shape(a)[0]
    # Compute the Gram matrix by taking the dot product of reshaped A with its transpose
    G = tf.matmul(a, a, transpose_a=True)
    # Returns the normalized Gram matrix 
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
            progress_bar.progress((i+1) / iterations)
        generated_images.append(generated.numpy())
        logging.info(f"Iteration {i+1}/{iterations} completed. Best cost: {best_cost:.2f}")
    logging.info(f"Training completed. Best cost: {best_cost:.2f}")
    return best_image



# Streamlit interface


def streamlit_interface():
    content_img = None
    style_img = None
    content_array = None
    style_array = None
    cols = st.columns(2)
    use_local_content = cols[0].checkbox("Use Local Content Image", value=False)
    use_local_style = cols[1].checkbox("Use Local Style Image", value=False)
    

    
    # Content Image Inputs
    if use_local_content:
        content_file = cols[0].file_uploader("Choose Content Image...", type=["jpg", "png", "jpeg"], key='content')
        if content_file:
            content_img = Image.open(content_file)
    else:
        content_url = cols[0].text_input("Enter Content Image URL", "")
        if content_url:
            content_img = Image.open(BytesIO(requests.get(content_url).content))
            
    if content_img:
        content_array = load_and_process_img(content_img)
        cols[0].image(content_img, caption='Content Image', use_column_width=True)
        
    
    # Style Image Inputs      
    if use_local_style:
        style_file = cols[1].file_uploader("Choose Style Image...", type=["jpg", "png", "jpeg"], key='style')
        if style_file:
            style_img = Image.open(style_file)    
    else:
        style_url = cols[1].text_input("Enter Style Image URL", "")
        if style_url:
            style_img = Image.open(BytesIO(requests.get(style_url).content))

    if style_img:
        style_array = load_and_process_img(style_img)
        cols[1].image(style_img, caption='Style Image', use_column_width=True)

    # Iterations slider
    iterations = st.slider("Number of iterations", min_value=10, max_value=500, value=20, step=10)
    
    # Style slider
    selected_style_layers_list = st.select_slider(
    'Select Style Layers List',
    options=list(style_layers_lists.keys()))

    selected_style_layers = style_layers_lists[selected_style_layers_list]
    
    return content_array, style_array, iterations, selected_style_layers






content_array, style_array, iterations, selected_style_layers = streamlit_interface()

if content_array is not None and style_array is not None:
    content_model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer(content_layer).output)
    style_models = [Model(inputs=vgg_model.input, outputs=vgg_model.get_layer(layer).output) for layer in selected_style_layers]

    

    if st.button("Generate"):
        with st.spinner('Processing...'):
            best_image = training_loop(content_array, style_array, iterations=iterations)
            display_image(best_image)



