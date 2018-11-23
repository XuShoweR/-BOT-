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

# PATH_TO_CKPT = "/usr/local/tensorflow/models/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb"
# PATH_TO_CKPT = "/home/fs168/Downloads/faster_rcnn_nas_coco_2018_01_28/frozen_inference_graph.pb"
# PATH_TO_CKPT = "/usr/local/tensorflow/models/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb"
# PATH_TO_CKPT = "/home/fs168/Document/final/frozen_inference_graph.pb"
PATH_TO_CKPT = "/home/fs168/Document/resnet/frozen_inference_graph.pb"
PATH_TO_CKPT = "/home/fs168/Document/resnet_02/frozen_inference_graph.pb"
PATH_TO_LABELS = "/home/fs168/NewRetail/data/person_label_map.pbtxt"
dir_path = '/home/fs168/NewRetail/src_dir/checkpoint_adam/'
# PATH_TO_LABELS = "/usr/local/tensorflow/models/research/object_detection/data/mscoco_label_map.pbtxt"
NUM_CLASSES = 1
ori_obj_dict = {"minx":0, "miny":0, "maxx":0, "maxy":0, "staff":0, "customer":0, "stand":0, "sit":0, "play_with_phone":0, "male":0, "female":0, "confidence":0.0}
json_path = './test.json'
# image_path = '/home/fs168/dataSet/challenge2018_test'
image_path = '/home/fs168/Public/project/tensorflow_detection/test_images'
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
# file = open(PATH_TO_LABELS, 'r')
# label_map = file.readlines()
# for i in range(80):
#     label_dict[label_map[i * 5 + 2].split(":")[1][:3].strip()] = label_map[i * 5 + 1].split(":")[1]
def max_num(x, y):
    if x > y:
        return x
    else:
        return y
time_start = time.time()
# per_image = start
# 开始检测
sess_cls = tf.Session()
saver = tf.train.import_meta_graph(dir_path + '-40.meta')
saver.restore(sess_cls, dir_path + '-40')
graph = tf.get_default_graph()

X_origin = tf.get_collection('X_origin')[0]
# X_imgages = read_img(test_img_path)
# start = time.time()
prediction = tf.get_collection('prediction')[0]

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        number = 0
        json_file = open(json_path, 'w')
        json_file.write('{\n')
        json_file.write('\t"results": [\n')
        list_file = os.listdir(image_path)
        count = 0
        ori = 0
        for i in range(1, 6):
            val_dir = '/home/fs168/dataSet/BOT新零售技术赛 测试集1 标注图片/scene_%d val' % i
            file_list = os.listdir(val_dir)
            for file_name in file_list:
                start = time.time()
                json_file.write('\t{\n')
                # image = Image.open(image_path +'/'+ img_file)
                id_num = int(file_name.split('_')[-1][:5])
                if i == 1:
                    image_id = "id_" + str(id_num)
                elif i == 2:
                    image_id = "id_" + str(id_num + 3520)
                elif i == 3:
                    image_id = "id_" + str(id_num + 5265)
                elif i == 4:
                    image_id = "id_" + str(id_num + 7113)
                else:
                    image_id = "id_" + str(id_num + 7769)
                file_path = os.path.join(val_dir, file_name)
                image = Image.open(file_path)
                image_np = np.array(image).astype(np.uint8)
                image_np_expanded = np.expand_dims(image_np , axis=0)
                # Actual detection.
                json_file.write('\t\t"image_id": "' + image_id +'",\n')
                json_file.write('\t\t"object": [\n')
                (boxes, scores, classes, num) = sess.run(
                  [detection_boxes, detection_scores, detection_classes, num_detections],
                  feed_dict={image_tensor: image_np_expanded})

                list = classes == 1
                person_boxes = boxes[list]
                person_scores = scores[list]
                person_classes = classes[list]
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
                # print(num)
                # cv2.imshow('img', image_np)
                # cv2.waitKey(0)


                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                im_width, im_height = image.size
                obj_list = []
                # predic_str = ""
                person_list = []
                num = len(np.where(person_scores > 0.30)[0])
                if num == 0:
                    json_file.write('\t]\n')
                    json_file.write('\t},\n')
                    continue
                for index in range(int(num)):
                    ymin, xmin, ymax, xmax = person_boxes[index][0:4]
                    (bottom, left, top, right) = int(ymin * im_height), int(xmin * im_width),\
                                                 int(ymax * im_height), int(xmax * im_width)
                    obj_dict = ori_obj_dict.copy()
                    obj_dict["miny"] = bottom
                    obj_dict["minx"] = left
                    obj_dict["maxx"] = right
                    obj_dict["maxy"] = top
                    if person_scores[index] < 0.4:
                        person_scores[index] += 0.6
                    elif person_scores[index] < 0.5:
                        person_scores[index] += 0.5
                    elif person_scores[index] < 0.7:
                        person_scores[index] += 0.3
                    elif person_scores[index] < 0.8:
                        person_scores[index] += 0.2
                    obj_dict["confidence"] = person_scores[index]
                    person_image = image_np[bottom:top, left:right]
                    max = max_num(right - left, top - bottom)
                    width = int((max - (right - left)) / 2)
                    height = int((max - (top - bottom)) / 2)
                    img_crop_person = cv2.copyMakeBorder(person_image, height, height, width, width,
                                                         cv2.BORDER_REPLICATE)
                    person_image = cv2.resize(img_crop_person, (112, 112))
                    person_list.append(person_image)
                    # put all object_dict of image to list
                    obj_list.append(obj_dict)

                # with tf.Session() as sess_cls:
                # saver = tf.train.import_meta_graph(dir_path + '-40.meta')
                # saver.restore(sess_cls, dir_path + '-40')
                # graph = tf.get_default_graph()
                #
                # X_origin = tf.get_collection('X_origin')[0]
                # # X_imgages = read_img(test_img_path)
                feed_dict = {X_origin: person_list}
                # start = time.time()
                # pred = tf.get_collection('prediction')[0]
                pred = sess_cls.run(prediction, feed_dict)
                result = np.ones(pred.shape, np.float32)
                result[pred <= 0] = 0
                person_index = 0
                attr_index = 0
                for result_attr in result:
                    if result_attr[0] == 1:
                        obj_list[person_index]['staff'] = 0
                        obj_list[person_index]['customer'] = 1
                    else:
                        obj_list[person_index]['staff'] = 1
                        obj_list[person_index]['customer'] = 0
                    if result_attr[1] == 1:
                        obj_list[person_index]['stand'] = 1
                        obj_list[person_index]['sit'] = 0
                    else:
                        obj_list[person_index]['stand'] = 0
                        obj_list[person_index]['sit'] = 1
                    if result_attr[2] == 1:
                        obj_list[person_index]['female'] = 1
                        obj_list[person_index]['male'] = 0
                    else:
                        obj_list[person_index]['female'] = 0
                        obj_list[person_index]['male'] = 1
                    if result_attr[3] == 1:
                        obj_list[person_index]['play_with_phone'] = 1
                    else:
                        obj_list[person_index]['play_with_phone'] = 0
                    json_file.write('\t{\n')
                    json_file.write('\t"minx": ' + str(obj_list[person_index]["minx"]) + ',\n')
                    json_file.write('\t"miny": ' + str(obj_list[person_index]["miny"]) + ',\n')
                    json_file.write('\t"maxx": ' + str(obj_list[person_index]["maxx"]) + ',\n')
                    json_file.write('\t"maxy": ' + str(obj_list[person_index]["maxy"]) + ',\n')
                    json_file.write('\t"staff": ' + str(obj_list[person_index]["staff"]) + ',\n')
                    json_file.write('\t"customer": ' + str(obj_list[person_index]["customer"]) + ',\n')
                    json_file.write('\t"stand": ' + str(obj_list[person_index]["stand"]) + ',\n')
                    json_file.write('\t"sit": ' + str(obj_list[person_index]["sit"]) + ',\n')
                    json_file.write('\t"play_with_phone": ' + str(obj_list[person_index]["play_with_phone"]) + ',\n')
                    json_file.write('\t"male": ' + str(obj_list[person_index]["male"]) + ',\n')
                    json_file.write('\t"female": ' + str(obj_list[person_index]["female"]) + ',\n')
                    json_file.write('\t"confidence": ' + str(obj_list[person_index]["confidence"]) + '\n')
                    attr_index += 1
                    person_index += 1
                    if attr_index < len(result):
                        json_file.write('\t},\n')
                    else:
                        json_file.write('\t}\n')
                json_file.write('\t]\n')
                json_file.write(('\t\t},\n'))
                end = time.time()
                print(end - start)
            # for bottom, left, top, right in locations:
            #     cv2.rectangle(image_np, (left, bottom), (right, top), color=(0, 0, 255))
            # cv2.imshow('img', image_np)
            # cv2.waitKey(0)

                # per_image = time.time() - per_image
                # print("per_image", per_image)
                # per_image = time.time()
                    # cv2.imwrite("./result/img_0%d" %i)
                    # cv2.imshow("show", image_np)
                # write_row = [img_file.split('.')[0], predic_str]
                # csv_writer.writerow(write_row)
                # print(write_row)
        json_file.write('\t]\n')
        json_file.write('}')
# print("end:", time.time() - start)
# csv_file.close()
#         print(time.time() - ori)
json_file.close()
time_end = time.time()
print(time_end - time_start)