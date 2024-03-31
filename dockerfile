FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set up time zone.
ENV TZ=UTC
COPY venv /app/venv

# RUN sh ~/venv/bin/activate

RUN apt update 

RUN apt install g++ -y
RUN apt install gcc -y
RUN apt install vim -y
RUN apt install pip -y

RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

RUN pip install opencv-python-headless \
numpy \
tqdm \
opencv-python \
torchmetrics \
einops \
torchgeometry \
kornia \
viser \
trimesh \
omegaconf \
pykdtree

RUN apt install libgl1 -y
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install ipdb
RUN apt install sudo
# RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

# Install system libraries required by OpenCV.
RUN chmod 777 /usr/local/lib/python3.10/dist-packages/
WORKDIR "/app"
CMD ["bash"]
