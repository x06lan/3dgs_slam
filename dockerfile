ARG UBUNTU_VERSION=22.04
ARG NVIDIA_CUDA_VERSION=11.8.0
ARG CUDA_ARCHITECTURES=86

FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as builder

# COLMAP version and CUDA architectures.
ENV QT_XCB_GL_INTEGRATION=xcb_egl
ENV DEBIAN_FRONTEND=noninteractive

# Set up time zone.
ENV TZ=UTC

WORKDIR "/build"

RUN apt update -y
RUN apt install -y git rsync unzip vim g++ wget pip sudo

# system packages 
RUN DEBIAN_FRONTEND=noninteractive apt install -y \
    python3 \
    python3-sdl2 \
    python3-tk \
    python3-pip \
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
    libxext6 \
    ninja-build \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev


# Build and install COLMAP.
COPY ./colmap ./colmap
RUN cd ./colmap && \
    mkdir build && \
    cd build && \
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
    -DCMAKE_INSTALL_PREFIX=/colmap_installed && \
    ninja install

# Install python packages
RUN pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 xformers --index-url https://download.pytorch.org/whl/cu118

COPY ./requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt 


RUN cd ./colmap/pycolmap && \
    python3 -m pip install .

# Install system libraries required by OpenCV.
RUN chmod 777 /usr/local/lib/python3.10/dist-packages/
WORKDIR "/app"
CMD ["bash"]

