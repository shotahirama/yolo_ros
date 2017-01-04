# yolo_ros

### First
```
git clone git@github.com:shotahirama/yolo_ros.git
cd yolo_ros
git submodule init
git submodule update
cd darknet
make
wget http://pjreddie.com/media/files/yolo.weights
cd ..
catkin build
```

#### If you use GPU, rewrite Makefile in darknet
```
GPU=1
CUDNN=1
OPENCV=1
DEBUG=0
```

### yolo_ros node start
```
roslaunch yolo_ros start_yolo.launch
```
