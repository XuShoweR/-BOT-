#! /usr/bin/python
import os
from PIL import Image

datas = open("mpii_gtbox.txt").readlines()

imgpath = "mpii/"
ann_dir = 'gtboxs/'
for data in datas:
    datasplit = datas.split('|')
    img_name = datasplit[0]
    im = Image.open(imgpath + img_name)
    width, height = im.size

    gts = datasplit[1:]
    # write in xml file
    if os.path.exists(ann_dir + os.path.dirname(img_name)):
        pass
    else:
        os.makedirs(ann_dir + os.path.dirname(img_name))
        os.mknod(ann_dir + img_name[:-4] + '.xml')
    xml_file = open((ann_dir + img_name[:-4] + '.xml'), 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>gtbox</folder>\n')
    xml_file.write('    <filename>' + img_name + '</filename>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + str(width) + '</width>\n')
    xml_file.write('        <height>' + str(height) + '</height>\n')
    xml_file.write('        <depth>3</depth>\n')
    xml_file.write('     </size>\n')

    # write the region of text on xml file
    for img_each_label in gts:
        spt = img_each_label.split(',')
        xml_file.write('    <object>\n')
        xml_file.write('        <name>'+ spt[4].strip() + '</name>\n')
        xml_file.write('        <pose>Unspecified</pose>\n')
        xml_file.write('        <truncated>0</truncated>\n')
        xml_file.write('        <difficult>0</difficult>\n')
        xml_file.write('        <bndbox>\n')
        xml_file.write('            <xmin>' + str(spt[0]) + '</xmin>\n')
        xml_file.write('            <ymin>' + str(spt[1]) + '</ymin>\n')
        xml_file.write('            <xmax>' + str(spt[2]) + '</xmax>\n')
        xml_file.write('            <ymax>' + str(spt[3]) + '</ymax>\n')
        xml_file.write('        </bndbox>\n')
        xml_file.write('    </object>\n')

    xml_file.write('</annotation>')
    xml_file.close() #

print('Done.')