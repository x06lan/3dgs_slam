# 3d gaussian slatting slam

ref https://github.com/WangFeng18/3d-gaussian-splatting

ref https://github.com/luigifreda/pyslam

ref https://github.com/LiheYoung/Depth-Anything

test splatter render

```bash
git clone git@github.com:colmap/colmap.git
docker build  -t 3dgs_slam:0.1.0 .
docker run -it --rm -e  "DISPLAY=$DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix -v ./:/app --privileged --gpus all 3dgs_slam:0.1.0 bash
python3 src/gaussian_cuda/setup.py install
export PYTHONPATH="${PYTHONPATH}:/app/src"
python3 src/splatting/splatter.py
python3 -m src.splatting.trainer
```

docker compose

```bash
git clone git@github.com:colmap/colmap.git
docker-compose up -d
docker exec -it 3dgs_slam bash

export PYTHONPATH="${PYTHONPATH}:/app/src"
python3 -m src.splatting.trainer
python3 src/main.py
```
