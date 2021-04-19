"""
视频逆缩放成原始尺寸
"""
import os
import glob
import PIL
from PIL import Image
import numpy as np

# Alibaba_test:
resize_dict = {
'000263': (1280, 720),'606335': (1280, 720),'606375': (1280, 720),'607276': (1280, 720),'621273': (1920, 1080),'621277': (1280, 720),'621603': (1280, 720),
'621703': (1280, 720),'621716': (1280, 720),'625084': (1280, 720),'625669': (1280, 720),'625670': (1280, 720),'625801': (1920, 1080),'625857': (1280, 720),
'626038': (1280, 720),'626137': (1280, 720),'626139': (1280, 720),'626140': (1280, 720),'628021': (1280, 720),'628098': (1280, 720),'628259': (1280, 720),
'629332': (1280, 720),'635677': (1920, 1080),'638543': (1920, 1080),'638558': (1920, 1080),'638572': (1920, 1080),'638574': (1280, 720),'638580': (1280, 720),
'638581': (1920, 1080),'638597': (1920, 1080),'639110': (1920, 1080),'639180': (1920, 1080),'639215': (1920, 1080),'639269': (1920, 1080),'639451': (1280, 720),
'639463': (1080, 1920),'639469': (1280, 720),'639479': (1920, 1080),'639516': (1920, 1080),'639536': (1920, 1080),'639613': (1280, 720),'639614': (1920, 810),
'639629': (1920, 1080),'640476': (1920, 1080),'640487': (1920, 1080),'640496': (1280, 720),'641427': (1104, 622),'749851': (1280, 720)}

base_dir = './submit/ali2021test-resnet101_all_l3+l4_ep0260'
output_dir = './submit/ali2021_results_l3+l4'

for video_name, (width, height) in resize_dict.items():
    print('video:', video_name)
    video_path = os.path.join(base_dir, video_name)
    image_paths = glob.glob(os.path.join(video_path, '*.png'))

    output_video_dir = os.path.join(output_dir, video_name)
    if not os.path.exists(output_video_dir):
        os.makedirs(output_video_dir)

    for image_path in image_paths:
        image_name = image_path.split(os.path.sep)[-1][:-4]
        img = Image.open(image_path)
        img = img.resize((width, height), Image.NEAREST)
        # print(np.shape(img))
        output_image_path = os.path.join(output_video_dir, image_name+'.png')
        img.save(output_image_path)