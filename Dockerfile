#
# Dockerfile for TensorFlow with Streamlit and GPU support
#

# NVIDIA's TensorFlow container image as base
FROM nvcr.io/nvidia/tensorflow:24.03-tf2-py3

# Set environment variable to prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        && \
    rm -rf /var/lib/apt/lists/*



# Set the working directory inside the container
WORKDIR /app

# copy the requirements file into the image
COPY ./requirements.txt /app/requirements.txt


# Install the dependencies
RUN pip install -r requirements.txt

# Expose the port where Streamlit will run
EXPOSE 8501

# Copy the app file to the image
COPY style_transfer_app.py /app


# Set the command to run Streamlit app
ENTRYPOINT ["streamlit", "run", "style_transfer_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Optional: Uncomment the line below to use a bash shell for debugging purposes
#ENTRYPOINT ["/bin/bash"]






