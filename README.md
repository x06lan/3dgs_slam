# 3d gaussian slatting slam

ref https://github.com/WangFeng18/3d-gaussian-splatting

ref https://github.com/luigifreda/pyslam

ref https://github.com/LiheYoung/Depth-Anything

test vo
```bash
docker build  -t 3dgs_slam:0.1.0 .
docker run -it --rm -e "DISPLAY=$DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix -v ./:/app --privileged --gpus all 3dgs_slam:0.1.0 bash
python3 ./src/main_vo.py
```

test splatter render
```bash
docker build  -t 3dgs_slam:0.1.0 .
docker run -it --rm -e  "DISPLAY=$DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix -v ./:/app --privileged --gpus all 3dgs_slam:0.1.0 bash
python3 src/gaussian_cuda/setup.py install
export PYTHONPATH="${PYTHONPATH}:/app/src"
python3 src/splatting/splatter.py
```

test depth estimator
1. create dataset/data dir and dataset/result dir
2. put ur demo images in dataset/data
3. download the pretrained model from https://huggingface.co/spaces/LiheYoung/Depth-Anything/blob/main/checkpoints/depth_anything_vits14.pth
4. move it into src/depth_estimator/depth_anything/checkpoints/
5. run on docker
```bash
docker build  -t 3dgs_slam:0.1.0 .
docker run -it --rm -e  "DISPLAY=$DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix -v ./:/app --privileged --gpus all 3dgs_slam:0.1.0 bash
python3 src/depth_estimator/run.py
```
