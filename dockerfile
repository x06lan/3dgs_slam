ARG UBUNTU_VERSION=22.04
ARG NVIDIA_CUDA_VERSION=11.8.0

FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as builder

# COLMAP version and CUDA architectures.
ARG CUDA_ARCHITECTURES=86
ENV QT_XCB_GL_INTEGRATION=xcb_egl
ENV DEBIAN_FRONTEND=noninteractive

# Set up time zone.
ENV TZ=UTC

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

RUN pip3 install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu118

# COLMAP dependencies
RUN apt-get install -y --no-install-recommends --no-install-suggests \
        cmake \
        ninja-build \
        build-essential \
        libboost-program-options-dev \
        libboost-filesystem-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libeigen3-dev \
        libflann-dev \
        libfreeimage-dev \
        libmetis-dev \
        libgoogle-glog-dev \
        libgtest-dev \
        libgmock-dev \
        libsqlite3-dev \
        libglew-dev \
        qtbase5-dev \
        libqt5opengl5-dev \
        libcgal-dev \
        libceres-dev

# git is required to run the COLMAP pybind build D:
RUN apt-get install -y --no-install-recommends --no-install-suggests \
    git

# Build and install COLMAP.
COPY ./colmap ./colmap
RUN cd ./colmap && \
    mkdir build && \
    cd build && \
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
        -DCMAKE_INSTALL_PREFIX=/colmap_installed && \
    ninja install

# Install python packages
COPY ./requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt 
RUN pip3 install torchmetrics einops
RUN cd ./colmap/pycolmap && \
    sed -i 's/\r$//' generate_stubs.sh && \
    python3 -m pip install .

# Install system libraries required by OpenCV.
RUN chmod 777 /usr/local/lib/python3.10/dist-packages/
WORKDIR "/app"
CMD ["bash"]

