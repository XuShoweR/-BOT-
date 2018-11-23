# for i in range(1, 6):
#     val_dir = '/home/fs168/dataSet/BOT新零售技术赛 测试集1 标注图片/scene_%d val' % i
#     print(val_dir)
# import numpy as np
# print(len(np.where(np.array([1, 2, 3, 0.1]) > 0.5)[0]))
import cv2
import matplotlib.pyplot as plt
path_dir = '/home/fs168/dataSet/BOT新零售技术赛 测试集1 标注图片/scene_1 val/scene_1_00071.jpg'
path_dir = '/home/fs168/dataSet/标注数据集-顾客及导购数据集/标注图片/scene_5 train/scene_5_00027.jpg'
image = cv2.imread(path_dir)
result = cv2.rectangle(image, (546, 311), (725, 393), (2, 0, 225), thickness=1)
plt.imshow(result)
plt.show()
# cv2.imshow("result", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
str1 = 'scene_1_00071.jpg'
result = str1.split('_')[-1][:5][0]
print("id_" + str(int(result)))
print(str(1.0))
# import os
# import keras
# from keras.datasets import cifar10