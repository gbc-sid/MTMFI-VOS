import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import os
import cv2

from lib.image import imwrite_indexed
from lib.utils import AverageMeter, interpolate
from .augmenter import ImageAugmenter
# Target Model标准
from .discriminator import Discriminator
# Target Model（SGD）
# from .discriminator_gd import Discriminator
from .seg_network_bise import SegNetwork

from time import time


class TargetObject:

    def __init__(self, obj_id, disc_params, **kwargs):
        self.object_id = obj_id
        self.discriminator = Discriminator(**disc_params)
        self.disc_layer = disc_params.layer
        self.start_frame = None
        self.start_mask = None
        self.index = -1

        for key, val in kwargs.items():
            setattr(self, key, val)

    def initialize(self, ft, mask):
        self.discriminator.init(ft[self.disc_layer], mask)

    def classify(self, ft):
        return self.discriminator.apply(ft)


class Tracker(nn.Module):

    def __init__(self, augmenter: ImageAugmenter, feature_extractor, disc_params_l3, disc_params_l4,
                 refiner: SegNetwork, device):

        super().__init__()

        self.augmenter = augmenter
        self.augment = augmenter.augment_first_frame
        self.disc_params_l3 = disc_params_l3
        self.disc_params_l4 = disc_params_l4
        self.feature_extractor = feature_extractor

        self.refiner = refiner
        for m in self.refiner.parameters():
            m.requires_grad_(False)
        self.refiner.eval()

        self.device = device

        self.first_frames = []
        self.current_frame = 0
        self.current_masks = None
        self.num_objects = 0

    def clear(self):
        self.first_frames = []
        self.current_frame = 0
        self.current_masks = None
        self.num_objects = 0
        torch.cuda.empty_cache()

    def run_dataset(self, dataset, out_path, speedrun=False, restart=None):
        """
        :param dataset:   Dataset to work with (See datasets.py)
        :param out_path:  Root path for storing label images. Sequences of label pngs will be created in subdirectories.
        :param speedrun:  [Optional] Whether or not to warm up Pytorch when measuring the run time. Default: False
        :param restart:   [Optional] Name of sequence to restart from. Useful for debugging. Default: None
        """
        self.visual_path = '/root/VOS/frtm-vos-otherlayer/visuallize'
        out_path.mkdir(exist_ok=True, parents=True)

        dset_fps = AverageMeter()

        print('Evaluating', dataset.name)
        restarted = False
        for sequence in dataset:
            if restart is not None and not restarted:
                if sequence.name != restart:
                    continue
                restarted = True
            # 测试可视化用
            # if sequence.name != "drift-straight":
            #     continue
            # 这个序列太大，暂时先不考虑
            if sequence.name == "226f1e10f7" \
                    or sequence.name == "1bcd8a65de" \
                    or sequence.name == "30fe0ed0ce" \
                    or sequence.name == "4bef684040" \
                    or sequence.name == "5f8241a147" \
                    or sequence.name == "b1a8a404ad":
                continue
            dst = out_path / sequence.name

            self.visual_path_new = self.visual_path + '/' + sequence.name
            if not os.path.exists(self.visual_path_new):
                os.makedirs(self.visual_path_new)


            # 预加载数据
            # We preload data as we cannot both read from disk and upload to the GPU in the background,
            # which would be a reasonable thing to do. However, in PyTorch, it is hard or impossible
            # to upload images to the GPU in a data loader running as a separate process.
            sequence.preload(self.device)
            self.clear()  # Mitigate out-of-memory that may occur on some YouTubeVOS sequences on 11GB devices.
            outputs, seq_fps = self.run_sequence(sequence, speedrun)
            dset_fps.update(seq_fps)

            dst.mkdir(exist_ok=True)
            for lb, f in zip(outputs, sequence.frame_names):
                imwrite_indexed(dst / (f + ".png"), lb)

        print("Average frame rate: %.2f fps" % dset_fps.avg)

    def run_sequence(self, sequence, speedrun=False):
        """
        :param sequence:  FileSequence to run.
        :param speedrun:  Only for DAVIS 2016: If True, let pytorch initialize its buffers in advance
                          to not incorrectly measure the memory allocation time in the first frame.
        :return:
        """

        self.eval()
        self.object_ids = sequence.obj_ids
        self.current_frame = 0
        self.targets_l3 = dict()
        self.targets_l4 = dict()

        N = 0

        object_ids = torch.tensor([0] + sequence.obj_ids, dtype=torch.uint8, device=self.device)  # Mask -> labels LUT

        if speedrun:
            image, labels, obj_ids = sequence[0]
            image = image.to(self.device)
            labels = labels.to(self.device)
            self.initialize(image, labels, sequence.obj_ids)  # Assume DAVIS 2016
            self.track(image)
            torch.cuda.synchronize()
            self.targets_l3 = dict()
            self.targets_l4 = dict()

        outputs = []
        t0 = time()
        self.num = 0
        for i, (image, labels, new_objects) in tqdm.tqdm(enumerate(sequence), desc=sequence.name, total=len(sequence),
                                                         unit='frames'):

            old_objects = set(self.targets_l4.keys())
            #print('old_object_l3:', set(self.targets_l3.keys()))
            #print('old_object_l4:', set(self.targets_l4.keys()))

            image = image.to(self.device)
            if len(new_objects) > 0:
                labels = labels.to(self.device)
                self.initialize(image, labels, new_objects)

            if len(old_objects) > 0:
                self.track(image)

                masks = self.current_masks
                # print("sequence.obj_ids:", len(sequence.obj_ids))
                if len(sequence.obj_ids) == 1:
                    labels = object_ids[(masks[1:2] > 0.5).long()]
                else:
                    masks = torch.clamp(masks, 1e-7, 1 - 1e-7)
                    masks[0:1] = torch.min((1 - masks[1:]), dim=0, keepdim=True)[0]  # background activation
                    segs = F.softmax(masks / (1 - masks), dim=0)  # s = one-hot encoded object activations
                    labels = object_ids[segs.argmax(dim=0)]

            if isinstance(labels, list) and len(labels) == 0:  # No objects yet
                labels = image.new_zeros(1, *image.shape[-2:])

            outputs.append(labels)
            self.current_frame += 1
            N += 1
            self.num += 1


        torch.cuda.synchronize()
        T = time() - t0
        fps = N / T

        return outputs, fps

    def initialize(self, image, labels, new_objects):

        self.current_masks = torch.zeros((len(self.targets_l3) + len(new_objects) + 1, *image.shape[-2:]),
                                         device=self.device)

        for obj_id in new_objects:
            # Create target

            mask = (labels == obj_id).byte()
            target_l3 = TargetObject(obj_id=obj_id, index=len(self.targets_l3) + 1, disc_params=self.disc_params_l3,
                                     start_frame=self.current_frame, start_mask=mask)
            target_l4 = TargetObject(obj_id=obj_id, index=len(self.targets_l4) + 1, disc_params=self.disc_params_l4,
                                     start_frame=self.current_frame, start_mask=mask)
            self.targets_l3[obj_id] = target_l3
            self.targets_l4[obj_id] = target_l4

            # HACK for debugging
            torch.random.manual_seed(0)
            np.random.seed(0)

            # Augment first image and extract features

            im, msk = self.augment(image, mask)
            with torch.no_grad():
                ft_l3 = self.feature_extractor(im, [target_l3.disc_layer])
                ft_l4 = self.feature_extractor(im, [target_l4.disc_layer])
            target_l3.initialize(ft_l3, msk)
            target_l4.initialize(ft_l4, msk)

            self.current_masks[target_l4.index] = mask

        return self.current_masks

    def track(self, image):
        with torch.no_grad():
            im_size = image.shape[-2:]
            features = self.feature_extractor(image)
        # Classify
        for (obj_id_l3, target_l3), (obj_id_l4, target_l4)\
                in zip(self.targets_l3.items(), self.targets_l4.items()):
            if target_l4.start_frame < self.current_frame:

                x_l3 = features[target_l3.disc_layer]
                x_l4 = features[target_l4.disc_layer]
                s_l3 = target_l3.classify(x_l3)
                #s_l3_in = interpolate(s_l3, x_l4[0, 0, :, :].size())
                #x_l4 = x_l4 * s_l3_in
                s_l4 = target_l4.classify(x_l4)
                # # Target model可视化
                # img_l3 = interpolate(s_l3, im_size)
                # img_l3 = np.array(img_l3.squeeze(0).permute(1, 2, 0).cpu())
                # img_l4 = interpolate(s_l4, im_size)
                # img_l4 = np.array(img_l4.squeeze(0).permute(1, 2, 0).cpu())
                # img_l3 = img_l3/np.max(img_l3)*255
                # img_l4 = img_l4 / np.max(img_l4) * 255
                # zero = np.zeros(np.shape(img_l3))
                # if obj_id_l3 == 1:
                #     img_l3_r = np.concatenate((zero, zero, img_l3), axis=2)
                #     img_l4_r = np.concatenate((zero, zero, img_l4), axis=2)
                #     img_l3, img_l4 = img_l3_r, img_l4_r
                # if obj_id_l3 == 2:
                #     img_l3_g = np.concatenate((img_l3, zero, zero), axis=2)
                #     img_l4_g = np.concatenate((img_l4, zero, zero), axis=2)
                #     img_l3 = img_l3_r + img_l3_g
                #     img_l4 = img_l3_r + img_l4_g
                # if obj_id_l3 == 3:
                #     img_l3_b = np.concatenate((zero, img_l3, zero), axis=2)
                #     img_l4_b = np.concatenate((zero, img_l4, zero), axis=2)
                #     img_l3 = img_l3_r + img_l3_g + img_l3_b
                #     img_l4 = img_l3_r + img_l4_g + img_l4_b
        #
                with torch.no_grad():
                    y = torch.sigmoid(self.refiner(s_l3, s_l4, features, im_size))
                self.current_masks[target_l4.index] = y
        # l3_path = self.visual_path_new + '/img_l3'
        # l4_path = self.visual_path_new + '/img_l4'
        # if not os.path.exists(l3_path):
        #     os.makedirs(l3_path)
        # if not os.path.exists(l4_path):
        #     os.makedirs(l4_path)
        # cv2.imwrite(l3_path +'/' +  str(self.num)
        #             + '.png', img_l3)
        # cv2.imwrite(l4_path +'/' + str(self.num)
        #             + '.png', img_l4)


        # Update

        for obj_id, t1 in self.targets_l4.items():
            if t1.start_frame < self.current_frame:
                for obj_id2, t2 in self.targets_l4.items():
                    if obj_id != obj_id2 and t2.start_frame == self.current_frame:
                        self.current_masks[t1.index] *= (1 - t2.start_mask.squeeze(0)).float()

        p = torch.clamp(self.current_masks, 1e-7, 1 - 1e-7)
        p[0:1] = torch.min((1 - p[1:]), dim=0, keepdim=True)[0]
        segs = F.softmax(p / (1 - p), dim=0)
        inds = segs.argmax(dim=0)

        # self.out_buffer = segs * F.one_hot(inds, segs.shape[0]).permute(2, 0, 1)
        for i in range(self.current_masks.shape[0]):
            self.current_masks[i] = segs[i] * (inds == i).float()

        for (obj_id, target_l3), (obj_id, target_l4) \
                in zip(self.targets_l3.items(), self.targets_l4.items()):
            if target_l4.start_frame < self.current_frame \
                    and self.disc_params_l4.update_filters:
                target_l4.discriminator.update(self.current_masks[target_l4.index]
                                               .unsqueeze(0).unsqueeze(0))
            if target_l3.start_frame < self.current_frame \
                    and self.disc_params_l3.update_filters:
                target_l3.discriminator.update(self.current_masks[target_l3.index]
                                               .unsqueeze(0).unsqueeze(0))

        # for obj_id, target_l3 in self.targets_l3.items():
        #     if target_l3.start_frame < self.current_frame and self.disc_params_l3.update_filters:
        #         target_l3.discriminator.update(self.current_masks[target_l3.index].unsqueeze(0).unsqueeze(0))

        return self.current_masks
