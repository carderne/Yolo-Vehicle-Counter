# Yolo Vehicle Counter
Forked from [guptavasu1213/Yolo-Vehicle-Counter](https://github.com/guptavasu1213/Yolo-Vehicle-Counter)

## Overview
You Only Look Once (YOLO) is a CNN architecture for performing real-time object detection. The algorithm applies a single neural network to the full image, and then divides the image into regions and predicts bounding boxes and probabilities for each region. For more detailed working of YOLO algorithm, please refer to the [YOLO paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf).

This project aims to count every vehicle (motorcycle, bus, car, cycle, truck, train) detected in the input video using YOLOv3 object-detection algorithm.

## Working
<p align="center">
  <img src="./example.gif">
</p>
As shown in the image above, when the vehicles in the frame are detected, they are counted. After getting detected once, the vehicles get tracked and do not get re-counted by the algorithm.

You may also notice that the vehicles will initially be detected and the counter increments, but for a few frames, the vehicle is not detected, and then it gets detected again. As the vehicles are tracked, the vehicles are not re-counted if they are counted once.


## Installation
Tested on Ubuntu 22.04 with Python 3.10.4 installed.

1. Install required OS packages:
```bash
sudo apt install -y python3 python3-pip ffmpeg libsm6 libxext6 libgl1-mesa-glx
```

2. Install this script and all requirements:
```bash
pip install -e .
```

3. Download the yolov3 weights:
```bash
wget https://pjreddie.com/media/files/yolov3.weights -P yolo-coco
```

## Usage
Simple example using provided video file:
```bash
yolo bridge.mp4 --output=bridge_tagged.mp4 --display
```

Full CLI:
```
Usage: yolo [OPTIONS] VIDEO

Arguments:
  VIDEO  Path to input video  [required]

Options:
  --model PATH              Path to Yolo model dir
  --output PATH             Path to save output if wanted
  --confidence FLOAT        Minimum probability to filter weak detections
  --threshold FLOAT         Threshold when applying non-maxima suppression
  --display / --no-display  Display tagged video while creating
  --use-gpu / --no-use-gpu  Accelerate with CUDA/GPU
```

## Implementation details
* The detections are performed on each frame by using YOLOv3 object detection algorithm and displayed on the screen with bounding boxes.
* The detections are filtered to keep all vehicles like motorcycle, bus, car, cycle, truck, train. The reason why trains are also counted is because sometimes, the longer vehicles like a bus, is detected as a train; therefore, the trains are also taken into account.
* The center of each box is taken as a reference point (denoted by a green dot when performing the detections) when track the vehicles.
* Also, in order to track the vehicles, the shortest distance to the center point is calculated for each vehicle in the last 10 frames.
* If `shortest distance < max(width, height) / 2`, then the vehicles is not counted in the current frame. Else, the vehicle is counted again. Usually, the direction in which the vehicle moves is bigger than the other one.
* For example, if a vehicle moves from North to South or South to North, the height of the vehicle is most likely going to be greater than or equal to the width. Therefore, in this case, `height/2` is compared to the shortest distance in the last 10 frames.
* As YOLO misses a few detections for a few consecutive frames, this issue can be resolved by saving the detections for the last 10 frames and comparing them to the current frame detections when required. The size of the vehicle does not vary too much in 10 frames and has been tested in multiple scenarios; therefore, 10 frames was chosen as an optimal value.

## Dependencies for using GPU for computations
1. Installing GPU appropriate drivers by following Step #2 in [this post](https://www.pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-ubuntu/).
2. Installing OpenCV for GPU computations: pip installable OpenCV does not support GPU computations for `dnn` module. Therefore, [this post](https://www.pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/) walks through installing OpenCV which can leverage the power of a GPU.

## Reference
* [YOLO object detection with OpenCV](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/)
