"""
视频缩放成480p
"""
import os
import glob
import PIL
from PIL import Image
import numpy as np

def image_resize(large_video_names, base_dir, output_dir):
    """

    :param large_video_names: 视频序列名称
    :param base_dir: 原始视频数据集路径
    :param output_dir: 缩放数据集路径
    :return:
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for video_name in large_video_names:
        print('video:', video_name)
        video_path = os.path.join(base_dir, video_name)
        image_paths = glob.glob(os.path.join(video_path, '*.jpg'))
        # print(os.path.join(video_path, '*.jpg'))
        output_video_dir = os.path.join(output_dir, video_name)
        # print(output_video_dir)
        if not os.path.exists(output_video_dir):
            # print('no')
            os.makedirs(output_video_dir)
        for image_path in image_paths:
            image_name = image_path.split(os.path.sep)[-1][:-4]
            # print(image_name)
            img = Image.open(image_path)
            print(np.shape(img))
            img.thumbnail((860, 480), Image.ANTIALIAS)
            output_image_path = os.path.join(output_video_dir, image_name + '.jpg')
            img.save(output_image_path)

def anno_resize(large_video_names, base_dir, output_dir):
    """

    :param large_video_names: 视频序列名称
    :param base_dir: 原始掩膜路径
    :param output_dir: 缩放掩膜路径
    :return:
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for video_name in large_video_names:
        print('video:', video_name)
        video_path = os.path.join(base_dir, video_name)
        image_paths = glob.glob(os.path.join(video_path, '*.png'))
        # print(os.path.join(video_path, '*.jpg'))
        output_video_dir = os.path.join(output_dir, video_name)
        # print(output_video_dir)
        if not os.path.exists(output_video_dir):
            # print('no')
            os.makedirs(output_video_dir)
        for image_path in image_paths:
            image_name = image_path.split(os.path.sep)[-1][:-4]
            print(image_name)
            img = Image.open(image_path)
            img.thumbnail((860, 480), Image.NEAREST)
            output_image_path = os.path.join(output_video_dir, image_name + '.png')
            img.save(output_image_path)

if __name__=='__main__':

    sequence = './data/PreRoundData/ImageSets/test.txt'
    large_video_names = []
    with open(sequence, "r") as lines:
        for line in lines:
            name = line.strip('\n')
            large_video_names.append(name)

    print("........starting images resize........")
    image_base_dir = './data/PreRoundData/JPEGImages'
    image_output_dir = './data/PreRoundData_480p/JPEGImages'
    image_resize(large_video_names, image_base_dir, image_output_dir)
    print("........finishing images resize........")

    print("........starting annos resize........")
    anno_base_dir = './code/PolarMask/results'
    anno_output_dir = './data/PreRoundData_test_480p/Annotations'
    anno_resize(large_video_names, anno_base_dir, anno_output_dir)
    print("........finishing annos resize........")