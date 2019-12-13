from glob import glob
import shutil
import os
import pdb

dst_jpg_dir = "data4YOLO/image_folder"
dst_voc_dir = "data4YOLO/annot_folder"

count = 0
for jpgfile in glob('trainval/*/*_image.jpg'):
    folder_path = os.path.dirname(os.path.abspath(jpgfile))
    # # Transfer the JPG File
    old_jpg_file_name = (os.path.split(jpgfile)[-1])
    # shutil.copy(jpgfile, dst_jpg_dir)

    # dst_jpg_file = os.path.join(dst_jpg_dir, old_jpg_file_name)
    # new_jpg_file_name = os.path.join(dst_jpg_dir, '{0:05}_image.jpg'.format(count))
    # os.rename(dst_jpg_file, new_jpg_file_name)

    # Transfer the VOC File
    old_voc_file_name = old_jpg_file_name.replace('_image.jpg', '_VOC.xml')
    vocfile = folder_path + '/' + old_voc_file_name
    shutil.copy(vocfile, dst_voc_dir)

    dst_voc_file = os.path.join(dst_voc_dir, old_voc_file_name)
    new_voc_file_name = os.path.join(dst_voc_dir, '{0:05}_VOC.xml'.format(count))
    # new_voc_file_name = os.path.join(dst_voc_dir, '{0:05}.xml'.format(count))
    os.rename(dst_voc_file, new_voc_file_name)

    count += 1