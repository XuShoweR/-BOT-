import cv2
import numpy as np
import os
import gc

file_dir = '/home/fs168/dataSet/crop_person'
# 0_101_102_110_121_142_421_720_745_836_840_883_920_41_img_00000052.jpg


def process_img(file_path):
    '''rough process images'''
    img = cv2.imread(file_path)

    width = int((300 - img.shape[1]) / 2)
    height = int((300 - img.shape[0]) / 2)
    img = cv2.copyMakeBorder(img, height, height, width, width, cv2.BORDER_REPLICATE)   # copyMakeBorder : fill new image border with original boerder
    img = cv2.resize(img, (112, 112))
    # img = np.multiply(img, 1/255.0)

    return img

file_list = os.listdir(file_dir)
np.random.shuffle(file_list)                                        # shuffle list

def get_data(file_list):
    images = []
    label = []
    count = 0
    for file_name in file_list:
        # processed_img = process_img(file_dir + '/' + file_name)
        processed_img = cv2.imread(file_dir + '/' + file_name)
        processed_img = cv2.resize(processed_img, (112, 112))
        file_parse_name = file_name.split('.')[0].split('_')

        label_class = np.zeros(4, np.int32)
        # label_attr = np.zeros(1001, np.int32)
        # class label
        cloth_category = file_parse_name[-1]
        for i in range(len(label_class)):
            label_class[i] = int(cloth_category[i])

        images.append(processed_img)
        label.append(label_class)
    return images, label
        # count += 1
    #     if count % 1000 == 0:
    #         print(count)
    #         np.save('/home/fs168/ClothesClassification/data/images_%d.npy' % (count / 1000), images)
    #         del images
    #         images = []
    #         gc.collect()
    # np.save('/home/fs168/ClothesClassification/data/images_%d.npy' % (count / 1000 + 1), images)
    # print(file_dir)