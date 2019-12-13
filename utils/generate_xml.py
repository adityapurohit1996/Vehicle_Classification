#! /usr/bin/python3

# Sample format
# <annotation>
#     <folder>_image_fashion</folder>
#     <filename>brooke-cagle-39574.jpg</filename>
#     <size>
#         <width>1200</width>
#         <height>800</height>
#         <depth>3</depth>
#     </size>
#     <segmented>0</segmented>
#     <object>
#         <name>head</name>
#         <pose>Unspecified</pose>
#         <truncated>0</truncated>
#         <difficult>0</difficult>
#         <bndbox>
#             <xmin>549</xmin>
#             <ymin>251</ymin>
#             <xmax>625</xmax>
#             <ymax>335</ymax>
#         </bndbox>
#     </object>
# <annotation>

import os
from glob import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import pdb

classes = (
    'Unknown', 'Compacts', 'Sedans', 'SUVs', 'Coupes',
    'Muscle', 'SportsClassics', 'Sports', 'Super', 'Motorcycles',
    'OffRoad', 'Industrial', 'Utility', 'Vans', 'Cycles',
    'Boats', 'Helicopters', 'Planes', 'Service', 'Emergency',
    'Military', 'Commercial', 'Trains'
)

def rot(n):
    n = np.asarray(n).flatten()
    assert(n.size == 3)

    theta = np.linalg.norm(n)
    if theta:
        n /= theta
        K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])

        return np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
    else:
        return np.identity(3)

def get_bbox(p0, p1):
    """
    Input:
    *   p0, p1
        (3)
        Corners of a bounding box represented in the body frame.

    Output:
    *   v
        (3, 8)
        Vertices of the bounding box represented in the body frame.
    *   e
        (2, 14)
        Edges of the bounding box. The first 2 edges indicate the `front` side
        of the box.
    """
    v = np.array([
        [p0[0], p0[0], p0[0], p0[0], p1[0], p1[0], p1[0], p1[0]],
        [p0[1], p0[1], p1[1], p1[1], p0[1], p0[1], p1[1], p1[1]],
        [p0[2], p1[2], p0[2], p1[2], p0[2], p1[2], p0[2], p1[2]]
    ])
    e = np.array([
        [2, 3, 0, 0, 3, 3, 0, 1, 2, 3, 4, 4, 7, 7],
        [7, 6, 1, 2, 1, 2, 4, 5, 6, 7, 5, 6, 5, 6]
    ], dtype=np.uint8)
    return v, e

files = glob('trainval/*/*_image.jpg')

DESTINATION_DIR = "converted_labels"

def create_root(file_prefix, width, height, folder_path, count):
    root = ET.Element("annotations")
    # ET.SubElement(root, "filename").text = "{}.jpg".format(file_prefix)
    temp_file_prefix = '{0:05}_image'.format(count)
    ET.SubElement(root, "filename").text = "{}.jpg".format(temp_file_prefix)
    ET.SubElement(root, "folder").text = '../data4YOLO/image_folder'
    # ET.SubElement(root, "folder").text = 'C:/Users/Thomas Kil/Documents/_Courses/Fall 2019/ROB 535/Final Project/_data/data4YOLO/image_folder'
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    return root


def create_object_annotation(root, voc_labels):
    for voc_label in voc_labels:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = voc_label[0]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = str(0)
        ET.SubElement(obj, "difficult").text = str(0)
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(voc_label[1])
        ET.SubElement(bbox, "ymin").text = str(voc_label[2])
        ET.SubElement(bbox, "xmax").text = str(voc_label[3])
        ET.SubElement(bbox, "ymax").text = str(voc_label[4])
    return root

def generate_voc_labels(file):
    voc_labels = []
    snapshot = file
    # print(snapshot)
    xyz = np.fromfile(snapshot.replace('_image.jpg', '_cloud.bin'), dtype=np.float32)
    xyz = xyz.reshape([3, -1])
    proj = np.fromfile(snapshot.replace('_image.jpg', '_proj.bin'), dtype=np.float32)
    proj.resize([3, 4])
    try:
        bbox = np.fromfile(snapshot.replace('_image.jpg', '_bbox.bin'), dtype=np.float32)
    except FileNotFoundError:
        print('[*] bbox not found.')
        bbox = np.array([], dtype=np.float32)
    bbox = bbox.reshape([-1, 11])
    uv = proj @ np.vstack([xyz, np.ones_like(xyz[0, :])])
    uv = uv / uv[2, :]
    for k, b in enumerate(bbox):
        R = rot(b[0:3])
        t = b[3:6]
        sz = b[6:9]
        vert_3D, edges = get_bbox(-sz / 2, sz / 2)
        vert_3D = R @ vert_3D + t[:, np.newaxis]

        vert_2D = proj @ np.vstack([vert_3D, np.ones(vert_3D.shape[1])])
        vert_2D = vert_2D / vert_2D[2, :]
        z_coords = vert_3D[2,:]
        if max(z_coords) < 50:
            x_coords = vert_2D[0,:]
            y_coords = vert_2D[1,:]
            min_x = max(0, (int)(min(x_coords)))
            max_x = min(1914, (int)(max(x_coords)))
            min_y = max(0, (int)(min(y_coords)))
            max_y = min(1052, (int)(max(y_coords)))
            c = classes[int(b[9])]

            if (c == "Compacts" or c == "Sedans" or
                c == "SUVs" or c == "Coupes" or c == "Muscle" or 
                c == "SportsClassics" or c == "Sports" or c == "Super"):
                pred_label = '1'
            elif (c == "Industrial" or c == "Utility" or c == "Vans" or
                  c == "Service" or c == "Emergency" or c == "Military" or
                  c == "Commercial"):
                pred_label = '2'
            elif (c == "Motorcycles" or c == "OffRoad" or c == "Cycles"):
                pred_label = '3'
            else:
                pred_label = '0'
            # voc_label = [c, min_x, min_y, max_x, max_y]
            voc_label = [pred_label, min_x, min_y, max_x, max_y]
            voc_labels.append(voc_label)
    return voc_labels

def create_file(folder_path, file_prefix, width, height, voc_labels, count):
    # pdb.set_trace()
    root = create_root(file_prefix, width, height, folder_path, count)
    root = create_object_annotation(root, voc_labels)
    tree = ET.ElementTree(root)
    file_name = file_prefix.split("_")[0] + '_VOC'
    tree.write("{}/{}.xml".format(folder_path, file_name))

if __name__ == "__main__":
    for i in range(len(files)):
    # for i in range(10):
        file = files[i]
        folder_path = os.path.dirname(os.path.abspath(file))
        # print(folder_path)
        image_name_jpg = (os.path.split(file)[-1])
        image_name = os.path.splitext(image_name_jpg)[0]
        voc_labels = generate_voc_labels(file)
        create_file(folder_path = folder_path, file_prefix=image_name, width=1914, height=1052, voc_labels=voc_labels, count = i)