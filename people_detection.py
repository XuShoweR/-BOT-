import numpy as np
import os
import tensorflow as tf
import cv2
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
from PIL import Image
import time
import math
import csv
from matplotlib import pyplot as plt

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

PATH_TO_CKPT = "/usr/local/tensorflow/models/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb"
PATH_TO_LABELS = "/usr/local/tensorflow/models/research/object_detection/data/mscoco_label_map.pbtxt"
NUM_CLASSES = 90
# image_path = "/root/PycharmProjects/root_tensorflow/tensorflow_detection/test_images/img_%02d.jpg"
# image_path = "/home/tensorflow/images/DataSets/2018-07-25/%06d.jpg"
# image_path = '/home/fs168/dataSet/challenge2018_test/0a0a615629231821.jpg'
# 0eb47f694d087f53.jpg fb39c40651e66069.jpg 8beb33e10d6cb201.jpg

image_path = '/home/fs168/dataSet/challenge2018_test'
csvfile_path = '/home/fs168/dataSet/test.csv'

# 加载模型进内存
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT , 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def , name='')
# 加载标签映射
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map , max_num_classes=NUM_CLASSES , use_display_name=True)
category_index = label_map_util.create_category_index(categories)
label_dict = {}
file = open(PATH_TO_LABELS, 'r')
label_map = file.readlines()
for i in range(80):
    label_dict[label_map[i * 5 + 2].split(":")[1][:3].strip()] = label_map[i * 5 + 1].split(":")[1]

start = time.time()
per_image = start
# 开始检测
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        number = 0
        csv_file = open(csvfile_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['ImageId', 'PredictionString'])
        list_file = os.listdir(image_path)
        # for i in range(141):
        for img_file in list_file:
            image = Image.open(image_path +'/'+ img_file)

            image_np = load_image_into_numpy_array(image)
            image_np = cv2.GaussianBlur(image_np, (3, 3), 1)
            image_np_expanded = np.expand_dims(image_np , axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
              [detection_boxes, detection_scores, detection_classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
            list = classes == 1
            # person_boxes = boxes[list]
            # person_scores = scores[list]
            # person_classes = classes[list]
            boxes = boxes[0]
            scores = scores[0]
            # vis_util.visualize_boxes_and_labels_on_image_array(
            #    image_np,
            #    np.squeeze(person_boxes),
            #    np.squeeze(person_classes).astype(np.int32),
            #    np.squeeze(person_scores),
            #    category_index,
            #    use_normalized_coordinates=True,
            #    line_thickness=3)
            # # plt.subplot(330 + i)
            # # plt.figure()
            # # plt.imshow(image_np)
            # cv2.imshow('img', image_np)
            # cv2.waitKey(0)
            # np.transpose()
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            im_width, im_height = image.size
            locations = []
            predic_str = ""
            write_row = []
            # label_maplist = label_map.split('}\nitem{')

            for index in range(int(num)):
                ymin, xmin, ymax, xmax = boxes[index][0:4]
                (bottom, left, top, right) = int(ymin * im_height), int(xmin * im_width),\
                                             int(ymax * im_height), int(xmax * im_width)
                # (bottom, left, top, right)
                # person_image = image_np[bottom:top, left:right]
                # print(top - bottom, right - left)
                # if person_image is not None and top - bottom > 50 and right - left > 50:
                if scores[index] > 0.3:
                    # cv2.imwrite("./person/%06d.jpg" % number, person_image)
                    # number += 1
                    flag = True
                    if index == 0:
                        locations.append((bottom, left, top, right))
                        predic_str = predic_str + "{} {} {} {} {} {} ".format(label_dict[str(int(classes[0][index]))].strip()[1:-1], scores[index], xmin, ymin, xmax, ymax)
                    else:
                        # filter
                        for IOU in range(len(locations)):
                            ori_ymin, ori_xmin, ori_ymax, ori_xmax = locations[IOU]
                            if math.fabs((right + left) / 2 - (ori_xmax + ori_xmin) / 2) < math.fabs((right - left) / 2 + (ori_xmax - ori_xmin) / 2):
                                overlap_area = (min(right, ori_xmax) - max(left, ori_xmin)) * ((min(top, ori_ymax)) - (max(bottom, ori_ymin)))
                                ori_area = (ori_xmax - ori_xmin) * (ori_ymax - ori_ymin)
                                new_area = (right - left) * (top - bottom)
                                if overlap_area / ori_area > 0.6 or overlap_area / new_area > 0.8:
                                    flag = False
                                    break
                        if flag:
                            locations.append((bottom, left, top, right))
                            predic_str = predic_str + "{} {} {} {} {} {} ".format(label_dict[str(int(classes[0][index]))].strip()[1:-1], scores[index], xmin, ymin, xmax, ymax)
            # for bottom, left, top, right in locations:
            #     cv2.rectangle(image_np, (left, bottom), (right, top), color=(0, 0, 255))
            # cv2.imshow('img', image_np)
            # cv2.waitKey(0)

            # per_image = time.time() - per_image
            # print("per_image", per_image)
            # per_image = time.time()
                # cv2.imwrite("./result/img_0%d" %i)
                # cv2.imshow("show", image_np)
            write_row = [img_file.split('.')[0], predic_str]
#             csv_writer.writerow(write_row)
#             print(write_row)
# print("end:", time.time() - start)
# csv_file.close()