#! /usr/bin/python3
from glob import glob
import csv
import os
import shutil

current_path = os.getcwd()
centroid_file_path = current_path + '/data/centroids/'
print(centroid_file_path)
count = 0
with open("centroids.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            # print(row[0])
            line_count += 1
        else:
            if line_count % 3 == 0:
                split_row = row[0].split('/')
                # FOR JPG
                jpg_file = (current_path + '/' + 'data/trainval/' +
                            split_row[0] + '/' + 
                            split_row[1] + '_image.jpg')
                shutil.copy(jpg_file, centroid_file_path + '/image_folder')
                orig_jpg_name = (centroid_file_path + '/image_folder/' + 
                                 split_row[1] + '_image.jpg')
                new_jpg_name = (centroid_file_path + '/image_folder/' + 
                                 split_row[0] + '_' + split_row[1] + 
                                 '_image.jpg')
                os.rename(orig_jpg_name, new_jpg_name)
                # FOR VOC
                voc_file = (current_path + '/' + 'data/trainval/' +
                            split_row[0] + '/' + 
                            split_row[1] + '_VOC.xml')
                shutil.copy(voc_file, centroid_file_path + '/annot_folder')
                orig_voc_name = (centroid_file_path + '/annot_folder/' + 
                                 split_row[1] + '_VOC.xml')
                new_voc_name = (centroid_file_path + '/annot_folder/' + 
                                 split_row[0] + '_' + split_row[1] + 
                                 '_VOC.xml')
                os.rename(orig_voc_name, new_voc_name)
            line_count += 1
        # count += 1
        # if count > 10:
        #     break
    print(f'Processed {line_count} lines.')
