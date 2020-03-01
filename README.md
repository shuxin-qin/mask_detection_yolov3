# mask_detection_yolov3
Detect face and mask using Yolov3 in pure tensorflow

### 1. Introduction

This is my face and mask detection using [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf) in pure TensorFlow. The yolov3 version is referred to the repo ( [YOLOv3_TensorFlow](https://github.com/wizyoung/YOLOv3_TensorFlow)  and  [tensorflow-yolov3](https://github.com/YunYang1994/tensorflow-yolov3) ). It contains the full pipeline of training and test on your own dataset. The key features of this repo are:

- Weights converter (converting pretrained darknet weights on COCO dataset to TensorFlow checkpoint.)
- Extremely fast GPU non maximum supression.
- Full training and testing pipeline.
- Kmeans algorithm to select prior anchor boxes.
- Trained weight with face mask dataset (can be downloaded [here](https://pan.baidu.com/s/1rQ8Hivjc3KH9u7XiI3yAXA). code: 0796) .

### 2. Requirements

Python version: 2 or 3

Packages:

- tensorflow >= 1.8.0
- opencv-python

### 3. Weights convertion

The pretrained darknet weights file can be downloaded [here](https://pjreddie.com/media/files/yolov3.weights). Place this weights file under directory `./data/darknet_weights/` and then run:

```shell
python convert_weight.py
```

Then the converted TensorFlow checkpoint file will be saved to `./data/darknet_weights/` directory.

You can also download the converted TensorFlow checkpoint file by me via [[Google Drive link](https://drive.google.com/drive/folders/1mXbNgNxyXPi7JNsnBaxEv1-nWr7SVoQt?usp=sharing)] or [[Github Release](https://github.com/wizyoung/YOLOv3_TensorFlow/releases/)] and then place it to the same directory.

### 4. Running

There are some demo images and videos under the `./data/demo_data/`. You can run the demo by:

Single image test demo:

Firstly, change some of the configuration information in image_test.py 

```shell
python image_test.py
```

Video test demo:

Firstly, change some of the configuration information in video_test.py 

```shell
python video_test.py
```

### 5. Training

#### 5.1 Data preparation 

(1) annotation file

Just use the xml format for annotation. The images and xml files are in the same folders.

Our Dataset class can read the annotations and images without any  additional operations.

(2)  class_names file:

The face mask dataset class names file is placed at `./data/face_mask.names`.

(3) prior anchor file:

Using the kmeans algorithm to get the prior anchors:

```
python cal_anchors.py
```

Then you will get 9 anchors and the average IoU. Save the anchors to a txt file.

(4) dataset:

Please refer to  [FaceMaskDetection](https://github.com/AIZOOTech/FaceMaskDetection).

#### 5.2 Training

Using `train_demo.py`. The hyper-parameters and the corresponding annotations can be found in cfgs.py:

```shell
python train_demo.py
```

Check the `cfgs.py` for more details. You should set the parameters yourself in your own specific task.

### 6. training tricks

Here are some training tricks from [YOLOv3_TensorFlow](https://github.com/wizyoung/YOLOv3_TensorFlow) ,:

Apply the two-stage training strategy or the one-stage training strategy:

(1) Two-stage training:

First stage: Restore `darknet53_body` part weights from COCO checkpoints, train the `yolov3_head` with big learning rate like 1e-3 until the loss reaches to a low level.

Second stage: Restore the weights from the first stage, then train the whole model with small learning rate like 1e-4 or smaller. At this stage remember to restore the optimizer parameters if you use optimizers like adam.

(2) One-stage training:

Just restore the whole weight file except the last three convolution layers (Conv_6, Conv_14, Conv_22). In this condition, be careful about the possible nan loss value.

-------

### Reference:

[YunYang1994/tensorflow-yolov3](https://github.com/YunYang1994/tensorflow-yolov3)

[YOLOv3_TensorFlow](https://github.com/wizyoung/YOLOv3_TensorFlow) 

[FaceMaskDetection](https://github.com/AIZOOTech/FaceMaskDetection)



