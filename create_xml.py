import cv2
import matplotlib.pyplot as plt
import json
import os

img_path = '/home/fs168/dataSet/标注数据集-顾客及导购数据集/标注图片/scene_1 train/scene_1_00001.jpg'
xml_dir = '/home/fs168/dataSet/Annotation'
ori_img_dir = '/home/fs168/dataSet/标注数据集-顾客及导购数据集/标注图片'
annotation_dir = '/home/fs168/dataSet/标注数据集-顾客及导购数据集/标注标签'
save_path = '/home/fs168/dataSet/crop_person'
mosaic_path = '/home/fs168/dataSet/masaike'

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
mosaic_list = os.listdir(mosaic_path)
for i in range(len(mosaic_list)):
    mosaic_list[i] = mosaic_list[i][:-9]

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
        # img = cv2.imread(img_dir_path)
        xml_name = xml_dir + '/' + annotation['filename'].split('.')[0] + '.xml'

        os.mknod(xml_name)
        xml_file = open(xml_name, 'w')
        xml_file.write('<annotation>\n')
        xml_file.write('    <folder>data</folder>\n')
        xml_file.write('    <filename>' + annotation['filename'] + '</filename>\n')
        xml_file.write('    <size>\n')
        xml_file.write('        <width>' + str(annotation['width']) + '</width>\n')
        xml_file.write('        <height>' + str(annotation['height']) + '</height>\n')
        xml_file.write('        <depth>3</depth>\n')
        xml_file.write('     </size>\n')
        index = 0
        for person in object:
            file_cmp_name = annotation['filename'][6:13]
            if (file_cmp_name + '_' + str(index) in mosaic_list):
                index += 1
                continue
            minx, maxx, miny, maxy = person['minx'], person['maxx'], person['miny'], person['maxy']
            xml_file.write('    <object>\n')
            xml_file.write('        <name>' + 'person' + '</name>\n')
            xml_file.write('        <pose>Unspecified</pose>\n')
            xml_file.write('        <truncated>0</truncated>\n')
            xml_file.write('        <difficult>0</difficult>\n')
            xml_file.write('        <bndbox>\n')
            xml_file.write('            <xmin>' + str(minx) + '</xmin>\n')
            xml_file.write('            <ymin>' + str(miny) + '</ymin>\n')
            xml_file.write('            <xmax>' + str(maxx) + '</xmax>\n')
            xml_file.write('            <ymax>' + str(maxy) + '</ymax>\n')
            xml_file.write('        </bndbox>\n')
            xml_file.write('    </object>\n')
            index += 1
        xml_file.write('</annotation>')
        xml_file.close()  #

    print('Done.')
        #     img_crop_person = []
        #     img_crop_person = img[miny:maxy, minx:maxx]
        #     save_file_name = file_name.split('.')[0][6:] + ('_%d' % index) + get_filename(person) + '.jpg'
        #     cv2.imwrite(save_path + '/' + save_file_name, img_crop_person)
        #     index += 1