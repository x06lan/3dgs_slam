FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set up time zone.
ENV TZ=UTC

# COPY venv /app/venv
# RUN sh ~/venv/bin/activate

WORKDIR "/build"

RUN apt update -y

RUN apt install -y rsync unzip vim g++ wget pip sudo

# system packages 
RUN DEBIAN_FRONTEND=noninteractive  apt install -y \
    python3 \
    python3-sdl2 \
    python3-tk \
    python3-pip \
    sudo \
    build-essential \
    cmake \
    libsuitesparse-dev \
    libprotobuf-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libpostproc-dev \
    libswscale-dev \
    libglew-dev \
    libeigen3-dev \
    libopencv-dev \
    python3-pyqt5 \
    libxcb-xinerama0 \
    libgl1 \
    ffmpeg \
    libsm6 \
    libxext6


# Install python packages

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118



COPY ./requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt 

# RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

# Install system libraries required by OpenCV.
RUN chmod 777 /usr/local/lib/python3.10/dist-packages/
WORKDIR "/app"
CMD ["bash"]
