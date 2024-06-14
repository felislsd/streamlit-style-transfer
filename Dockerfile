#
# This example Dockerfile illustrates a method to install
# additional packages on top of NVIDIA's TensorFlow container image.
#
# To use this Dockerfile, use the `docker build` command.
# See https://docs.docker.com/engine/reference/builder/
# for more information.
#
FROM nvcr.io/nvidia/tensorflow:24.03-tf2-py3

# Set environment variable to prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install NVIDIA drivers and OpenGL libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        && \
    rm -rf /var/lib/apt/lists/*


# copy the requirements file into the image
COPY ./requirements.txt /app/requirements.txt

# Create a directory for your app
WORKDIR /app





# Install the dependencies
RUN pip install -r requirements.txt

EXPOSE 8501

# copy every content from the local file to the image
COPY . /app

#HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Set the ENTRYPOINT to your Python script
ENTRYPOINT ["streamlit", "run", "style_transfer_app.py", "--server.port=8501", "--server.address=0.0.0.0"]






