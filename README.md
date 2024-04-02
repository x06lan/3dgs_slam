# 3d gaussian slatting slam

ref https://github.com/WangFeng18/3d-gaussian-splatting

ref https://github.com/luigifreda/pyslam


```bash
docker build  -t 3dgs_slam:0.1.0 .
docker run -it -rm -e "DISPLAY=$DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix -v ./:/app --privileged --gpus all 3dgs_slam:0.1.0 bash
python3 ./src/main_vo.py
```