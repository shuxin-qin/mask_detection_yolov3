# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 13:05:55 2020

@author: Qin
"""

import numpy as np 
import cv2
from utils.dataset import Dataset
import cfgs

dataset = Dataset('../dataset/face_mask/train', cfgs.classes, cfgs.img_size, cfgs.anchors, 'train', batch_size=8, multi_scale=True)

img_idx_batch, img_batch, y_true_13_batch, y_true_26_batch, y_true_52_batch = next(dataset)
'''
for i in range(4):
    j = 0
    for d in dataset:
        img_idx_batch, img_batch, y_true_13_batch, y_true_26_batch, y_true_52_batch = d
        print(i, j, img_batch[0].shape[0], img_idx_batch)
        j += 1
'''

image = img_batch[0]
image = image * 255
image = image.astype(np.uint8)
print(image.shape)

y_true = [y_true_13_batch, y_true_26_batch, y_true_52_batch]
num_images = y_true[0].shape[0]  # 检查的图片数量 Batch_size(8)
num_classes = y_true[0][0][..., 5:].shape[-1]
print('num_images: ', num_images)
print('num_classes: ', num_classes)

#batch中第0个样本
true_labels_list, true_boxes_list = [], []
for j in range(3):  # three feature maps 3个feature map
    # y_true : [feature_map_1(Batch_size,....), .....]
    true_probs_temp = y_true[j][0][..., 5:-1]  # 各个类别预测的概率
    true_boxes_temp = y_true[j][0][..., 0:4]  # boxes信息

    # 去除y_true中没有目标的anchor
    object_mask = true_probs_temp.sum(axis=-1) > 0

    # 取出feature_map中,只含有目标的单元
    true_probs_temp = true_probs_temp[object_mask]  # shape(13x13x3,class_id)
    true_boxes_temp = true_boxes_temp[object_mask]  # shape(13x13x3,boxes)

    true_labels_list += np.argmax(true_probs_temp, axis=-1).tolist()  # 存在目标的cell中,
    true_boxes_list += true_boxes_temp.tolist()

# 所有有效的存在真实值的 boxes
true_boxes = np.array(true_boxes_list)
box_centers, box_sizes = true_boxes[:, 0:2], true_boxes[:, 2:4]

# 坐标转换
true_boxes[:, 0:2] = box_centers - box_sizes / 2.  # 左上角坐标
true_boxes[:, 2:4] = true_boxes[:, 0:2] + box_sizes  # 右下角坐标

print(true_boxes.shape)

for coord in true_boxes:
    x, y, w, h = coord[0], coord[1], coord[2], coord[3]
    print(x, y, w, h)
    image = cv2.rectangle(image, (int(x), int(y)), (int(w), int(h)), (255, 0, 0), 2)    

cv2.imshow('image', image[..., ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()
