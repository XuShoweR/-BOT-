import cv2
import matplotlib.pyplot as plt
import json
import os

img_path = '/home/fs168/dataSet/标注数据集-顾客及导购数据集/标注图片/scene_1 train/scene_1_00001.jpg'
ori_img_dir = '/home/fs168/dataSet/标注数据集-顾客及导购数据集/标注图片'
annotation_dir = '/home/fs168/dataSet/标注数据集-顾客及导购数据集/标注标签'
save_path = '/home/fs168/dataSet/crop_person'

def get_filename(person_dict):
    suffix = ''
    if person_dict['customer'] == 1:
        suffix += '_1'
    else:
        suffix += '_0'
    suffix = suffix + str(person_dict['stand']) + str(person['gender']) + str(person['play_with_phone'])
    return suffix

def max_num(x, y):
    if x > y:
        return x
    else:
        return y

annotation_list = os.listdir(annotation_dir)
for scene_list in annotation_list:
    img_scene_name = scene_list[6:]
    file_list = os.listdir(annotation_dir + '/' + scene_list)
    for file_name in file_list:
        img_dir_path = ori_img_dir + '/' + img_scene_name + '/' + file_name.split('.')[0] + '.jpg'
        json_file_name = annotation_dir + '/' + scene_list + '/' + file_name
        f = open(json_file_name, encoding='utf-8')
        # img_name =
        hjson = json.load(f)
        annotation = hjson['annotation']
        annotation = annotation[0]
        object = annotation['object']
        index = 0
        img = cv2.imread(img_dir_path)
        for person in object:
            minx, maxx, miny, maxy = person['minx'], person['maxx'], person['miny'], person['maxy']
            img_crop_person = []
            img_crop_person = img[miny:maxy, minx:maxx]
            # if (maxx - minx) > (maxy - miny):
            #     center = (miny + maxy) / 2
            #     length = (maxx - minx)
            #     start = int(center - length / 2)
            #     end = start + length
            #     img_crop_person = img[start:end, minx:maxx]
            # else:
            #     center = (minx + maxx) / 2
            #     length = (maxy - miny)
            #     start = int(center - length / 2)
            #     end = start + length
            #     test = img.shape[1]
            #     img_crop_person = img[miny:maxy, start:end]
            max = max_num(img_crop_person.shape[1], img_crop_person.shape[0])
            width = int((max - img_crop_person.shape[1]) / 2)
            height = int((max - img_crop_person.shape[0]) / 2)
            img_crop_person = cv2.copyMakeBorder(img_crop_person, height, height, width, width,
                                     cv2.BORDER_REPLICATE)  # copyMakeBorder : fill new image border with original boerder
            # img = cv2.resize(img, (112, 112))
            # cv2.imshow('crop_%d' % index, img_crop_person)
            # cv2.waitKey(0)
            # cv2.destroyWindow()
            save_file_name = file_name.split('.')[0][6:] + ('_%d'  % index) + get_filename(person) + '.jpg'
            cv2.imwrite(save_path + '/' + save_file_name, img_crop_person)
            index += 1



# class people(object):
#     def __init__(self, people):
#         if people['customer'] == 1:
#             self.person_profession = 0
#         else:
#             self.person_profession = 1
#         self.play_with_phone = people['play_with_phone']
#         self.gender = people['gender']
#         self.sit = people['sit']
#         self.minx = people['minx']
#         self.maxx = people['maxx']