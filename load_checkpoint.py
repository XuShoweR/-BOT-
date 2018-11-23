import os
import cv2
import time
import numpy as np
import tensorflow as tf

# dict_fashion ={0:'T-shirt/top',1:'Trouser',2:'Pullover', 3:'Dress', 4:'Coat', 5:'Sandal', 6:'Shirt', 7:'Sneaker', 8:'Bag', 9:'Ankle boot'}
# dir_path = '/home/fs168/PycharmProjects/checkpoint_adam/'
dir_path = '/home/fs168/NewRetail/src_dir/checkpoint_adam/'
# test_img_path = '/home/fs168/PycharmProjects/test_imgs_person'
# test_img_path = '/home/fs168/NewRetail/src_dir/test_image'
test_img_path = '/home/fs168/dataSet/crop_person'

def read_img(filename):
    '''process img BGR to GRAY  & resize'''
    img_list = os.listdir(filename)
    img_output = []
    for img in img_list:
        img_process = cv2.imread(filename + '/' + img)
        img_process = cv2.resize(img_process, (112, 112))
        # img_process = -(img_process - 255)
        # print(img)
        # cv2.imshow('img', img_process)
        # cv2.waitKey(0)
        # img_process = np.reshape(img_process, [784])
        img_output.append(img_process)
    return img_output


with tf.Session() as sess:
    if os.path.exists(dir_path) is True:
        saver = tf.train.import_meta_graph(dir_path + '-40.meta')
        saver.restore(sess, dir_path + '-40')
        graph = tf.get_default_graph()

        X_origin = tf.get_collection('X_origin')[0]
        img_list = os.listdir(test_img_path)
        acc = 0
        for img in img_list:
            img_out = []
            # X_imgages = read_img(test_img_path + "/" + img)
            X_imgages = cv2.imread(test_img_path + "/" + img)
            X_imgages = cv2.resize(X_imgages, (112, 112))
            img_out.append(X_imgages)
            feed_dict = {X_origin:img_out}
            start = time.time()
            pred = tf.get_collection('prediction')[0]
            prediction = sess.run(pred, feed_dict)
            end = time.time()
            print(prediction.shape)
            result = np.ones(prediction.shape, np.float32)
            result[prediction <= 0] = 0
            for i in range(0, 4):
                if img.split('_')[-1][:4][i] == str(int(result[0][i])):
                    acc += 1
            person_index = 0
        print(acc / (4 * len(img_list)))
#         for result_attr in result:
#             obj_list = {}
#             if result_attr[0] == 1:
#                 obj_list['staff'] = 0
#                 obj_list['customer'] = 1
#             else:
#                 obj_list['staff'] = 1
#                 obj_list['customer'] = 0
#             if result_attr[1] == 1:
#                 obj_list['stand'] = 1
#                 obj_list['sit'] = 0
#             else:
#                 obj_list['stand'] = 0
#                 obj_list['sit'] = 1
#             if result_attr[2] == 1:
#                 obj_list['female'] = 1
#                 obj_list['male'] = 0
#             else:
#                 obj_list['female'] = 0
#                 obj_list['male'] = 1
#             if result_attr[3] == 1:
#                 obj_list['play_with_phone'] = 1
#             else:
#                 obj_list['play_with_phone'] = 0
#             print(obj_list)
# final = time.time()