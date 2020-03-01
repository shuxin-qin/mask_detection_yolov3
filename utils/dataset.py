#! /usr/bin/env python
# coding=utf-8

import os
import cv2
import random
import numpy as np
import xml.etree.ElementTree as ET
from utils.data_aug import *

class Dataset(object):
    """implement Dataset here"""
    def __init__(self, annot_path, classes, img_size, anchors, mode, batch_size=4,
                 multi_scale=False, letterbox_resize=True, interval=10):
        
        self.annot_path = annot_path
        self.class_num = len(classes)
        self.img_size = img_size
        self.anchors = anchors
        self.mode = mode
        self.batch_size = batch_size
        self.multi_scale = multi_scale
        self.mix_up = mix_up
        self.letterbox_resize = letterbox_resize
        self.interval = interval

        self.label_map = {}
        for i, val in classes.items():
            self.label_map[val] = i
               
        self.annotations = self.load_annotations_xml(self.annot_path)
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

    def __iter__(self):
        return self

    def __next__(self):

        if self.multi_scale and self.mode == 'train':
            random_img_size = [[x * 32, x * 32] for x in range(10, 20)]
            self.img_size = random.choice(random_img_size)

        img_idx_batch, img_batch, y_true_13_batch, y_true_26_batch, y_true_52_batch = [], [], [], [], []

        #if self.batch_count >= self.num_batchs:
        #    self.batch_count = 0
        num = 0
        if self.batch_count < self.num_batchs:
            while num < self.batch_size:
                index = self.batch_count * self.batch_size + num
                if index >= self.num_samples: 
                    index -= self.num_samples
                
                annotation = self.annotations[index]
                img_idx, img, y_true_13, y_true_26, y_true_52 = self.parse_data(
                    annotation, self.class_num, self.img_size, self.anchors, self.mode, self.letterbox_resize)

                img_idx_batch.append(img_idx)
                img_batch.append(img)
                y_true_13_batch.append(y_true_13)
                y_true_26_batch.append(y_true_26)
                y_true_52_batch.append(y_true_52)

                num += 1
            self.batch_count += 1             
            img_idx_batch = np.asarray(img_idx_batch, np.int64)
            img_batch = np.asarray(img_batch)
            y_true_13_batch = np.asarray(y_true_13_batch)
            y_true_26_batch = np.asarray(y_true_26_batch)
            y_true_52_batch = np.asarray(y_true_52_batch)
            return img_idx_batch, img_batch, y_true_13_batch, y_true_26_batch, y_true_52_batch
        else:
            self.batch_count = 0
            np.random.shuffle(self.annotations)
            raise StopIteration

    def __len__(self):
        return self.num_batchs

    def load_annotations(self, annot_path):
        with open(annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)
        return annotations

    def load_annotations_xml(self, annot_path, ext=['xml']):
        file_names = os.listdir(annot_path)
        annotations = []
        for filename in file_names:
            if filename.split('.')[-1] in ext:
                img_name = os.path.join(annot_path, filename.replace('.xml', '.jpg'))
                coords = self.parse_xml(os.path.join(annot_path, filename))
                if len(coords) == 0:
                    continue
                for coord in coords:
                    img_name += ' '+str(coord[0])+' '+str(coord[1])+' '+str(coord[2])+' '+str(coord[3])+' '+str(coord[4])
                annotations.append(img_name)

        np.random.shuffle(annotations)
        return annotations
                
    # 从xml文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, name]]
    def parse_xml(self, xml_path):
        '''
        输入：
            xml_path: xml的文件路径
        输出：
            从xml文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, label]]
        '''
        tree = ET.parse(xml_path)       
        root = tree.getroot()
        objs = root.findall('object')
        coords = list()
        for ix, obj in enumerate(objs):
            name = obj.find('name').text
            box = obj.find('bndbox')
            x_min = int(box[0].text)
            y_min = int(box[1].text)
            x_max = int(box[2].text)
            y_max = int(box[3].text)
            if name == 'face_nask':
                name = 'face_mask'
            label = self.label_map[name]
            coords.append([x_min, y_min, x_max, y_max, label])
        return coords

    def parse_data(self, line, class_num, img_size, anchors, mode, letterbox_resize):

        img_idx, pic_path, boxes, labels  = self.parse_line(line)
        img = cv2.imread(pic_path)
        # expand the 2nd dimension, mix up weight default to 1.
        boxes = np.concatenate((boxes, np.full(shape=(boxes.shape[0], 1), fill_value=1., dtype=np.float32)), axis=-1)

        if mode == 'train':
            # random color jittering
            # NOTE: applying color distort may lead to bad performance sometimes
            img = random_color_distort(img)
            # random expansion with prob 0.5
            if np.random.uniform(0, 1) > 0.5:
                img, boxes = random_expand(img, boxes, 4)

            # random cropping
            h, w, _ = img.shape
            boxes, crop = random_crop_with_constraints(boxes, (w, h))
            x0, y0, w, h = crop
            img = img[y0: y0+h, x0: x0+w]

            # resize with random interpolation
            h, w, _ = img.shape
            interp = np.random.randint(0, 5)
            img, boxes = resize_with_bbox(img, boxes, img_size[0], img_size[1], interp=interp, letterbox=letterbox_resize)

            # random horizontal flip
            h, w, _ = img.shape
            img, boxes = random_flip(img, boxes, px=0.5)
        else:
            img, boxes = resize_with_bbox(img, boxes, img_size[0], img_size[1], interp=1, letterbox=letterbox_resize)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        # the input of yolo_v3 should be in range 0~1
        img = img / 255.
        y_true_13, y_true_26, y_true_52 = self.process_box(boxes, labels, img_size, class_num, anchors)

        return img_idx, img, y_true_13, y_true_26, y_true_52


    def parse_line(self, line):

        s = line.strip().split(' ')
        assert len(s) > 5, 'Annotation error! Please check your annotation file. Make sure there is at least one target object in each image.'
        try:
            line_idx = int(s[0].split('-')[-1].split('.')[0])
        except:
            line_idx = 0
        pic_path = s[0]
        s = s[1:]
        assert len(s) % 5 == 0, 'Annotation error! Please check your annotation file. Maybe partially missing some coordinates?'
        box_cnt = len(s) // 5
        boxes = []
        labels = []
        for i in range(box_cnt):
            label, x_min, y_min, x_max, y_max = int(s[i*5+4]), float(s[i*5+0]), float(s[i*5+1]), float(s[i*5+2]), float(s[i*5+3])
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(label)
        boxes = np.asarray(boxes, np.float32)
        labels = np.asarray(labels, np.int64)
        return line_idx, pic_path, boxes, labels


    def process_box(self, boxes, labels, img_size, class_num, anchors):
        '''
        Generate the y_true label, i.e. the ground truth feature_maps in 3 different scales.
        params:
            boxes: [N, 5] shape, float32 dtype. `x_min, y_min, x_max, y_mix, mixup_weight`.
            labels: [N] shape, int64 dtype.
            class_num: int64 num.
            anchors: [9, 4] shape, float32 dtype.
        '''
        anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

        # convert boxes form:
        # shape: [N, 2]
        # (x_center, y_center)
        box_centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
        # (width, height)
        box_sizes = boxes[:, 2:4] - boxes[:, 0:2]

        # [13, 13, 3, 5+num_class+1] `5` means coords and labels. `1` means mix up weight. 
        y_true_13 = np.zeros((img_size[1] // 32, img_size[0] // 32, 3, 6 + class_num), np.float32)
        y_true_26 = np.zeros((img_size[1] // 16, img_size[0] // 16, 3, 6 + class_num), np.float32)
        y_true_52 = np.zeros((img_size[1] // 8, img_size[0] // 8, 3, 6 + class_num), np.float32)

        # mix up weight default to 1.
        y_true_13[..., -1] = 1.
        y_true_26[..., -1] = 1.
        y_true_52[..., -1] = 1.

        y_true = [y_true_13, y_true_26, y_true_52]

        # [N, 1, 2]
        box_sizes = np.expand_dims(box_sizes, 1)
        # broadcast tricks
        # [N, 1, 2] & [9, 2] ==> [N, 9, 2]
        mins = np.maximum(- box_sizes / 2, - anchors / 2)
        maxs = np.minimum(box_sizes / 2, anchors / 2)
        # [N, 9, 2]
        whs = maxs - mins

        # [N, 9]
        iou = (whs[:,:,0]*whs[:,:,1])/(box_sizes[:,:,0]*box_sizes[:,:,1]+anchors[:,0]*anchors[:,1]-whs[:,:,0]*whs[:,:,1]+1e-10)
        # [N]
        best_match_idx = np.argmax(iou, axis=1)

        ratio_dict = {1.: 8., 2.: 16., 3.: 32.}
        for i, idx in enumerate(best_match_idx):
            # idx: 0,1,2 ==> 2; 3,4,5 ==> 1; 6,7,8 ==> 0
            feature_map_group = 2 - idx // 3
            # scale ratio: 0,1,2 ==> 8; 3,4,5 ==> 16; 6,7,8 ==> 32
            ratio = ratio_dict[np.ceil((idx + 1) / 3.)]
            x = int(np.floor(box_centers[i, 0] / ratio))
            y = int(np.floor(box_centers[i, 1] / ratio))
            k = anchors_mask[feature_map_group].index(idx)
            c = labels[i]
            # print(feature_map_group, '|', y,x,k,c)

            y_true[feature_map_group][y, x, k, :2] = box_centers[i]
            y_true[feature_map_group][y, x, k, 2:4] = box_sizes[i]
            y_true[feature_map_group][y, x, k, 4] = 1.
            y_true[feature_map_group][y, x, k, 5 + c] = 1.
            y_true[feature_map_group][y, x, k, -1] = boxes[i, -1]

        return y_true_13, y_true_26, y_true_52
