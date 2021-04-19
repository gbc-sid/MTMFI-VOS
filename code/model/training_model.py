import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

from lib.utils import AverageMeter, interpolate, adaptive_cat

from torch.autograd import Variable
from lib.training_datasets import SampleSpec
from .augmenter import ImageAugmenter
# Target Model标准
from .discriminator import Discriminator


class TargetObject:
    def __init__(self, disc_params, **kwargs):

        self.discriminator = Discriminator(**disc_params)
        # **kwargs表示关键字参数，它本质上是一个dict；
        for key, val in kwargs.items():
            # setattr()设置属性值
            setattr(self, key, val)

    def initialize(self, ft, mask):
        self.discriminator.init(ft[self.discriminator.layer], mask)
    # 初始化（存在tmodel缓存）
    def initialize_pretrained(self, state_dict):
        self.discriminator.load_state_dict(state_dict)
        for p in self.discriminator.parameters():
            p.requires_grad_(False)
        self.discriminator.eval()

    def get_state_dict(self):
        return self.discriminator.state_dict()

    def classify(self, ft):
        return self.discriminator.apply(ft)


class TrainerModel(nn.Module):

    def __init__(self, augmenter, feature_extractor,
                 disc_params_l3, disc_params_l4,
                 seg_network, batch_size=0,
                 tmodel_cache_l3=None, tmodel_cache_l4=None,
                 device=None):

        super().__init__()

        self.augmenter = augmenter
        # First frame
        self.augment = augmenter.augment_first_frame
        # Target model?
        self.tmodels_l3 = [TargetObject(disc_params_l3) for _ in range(batch_size)]
        self.tmodels_l4 = [TargetObject(disc_params_l4) for _ in range(batch_size)]
        # F
        self.feature_extractor = feature_extractor
        # S
        self.refiner = seg_network
        # memory
        self.tmodel_cache_l3 = tmodel_cache_l3
        self.tmodel_cache_l4 = tmodel_cache_l4
        self.device = device
        # loss交叉熵
        self.compute_loss = nn.BCELoss()# nn.BCEWithLogitsLoss(), focal_loss()
        # 评测指标IoU
        self.compute_accuracy = self.intersection_over_union

        self.scores = None
        self.ft_channels = None

    def load_state_dict(self, state_dict):

        prefix = "refiner."
        sd2 = dict()
        for k, v in state_dict.items():
            assert k.startswith(prefix)
            k = k[len(prefix):]
            sd2[k] = v

        self.refiner.load_state_dict(sd2)

    def state_dict(self):
        return self.refiner.state_dict(prefix="refiner.")

    # count IoU
    def intersection_over_union(self, pred, gt):

        pred = (pred > 0.5).float()
        gt = (gt > 0.5).float()

        intersection = pred * gt
        i = intersection.sum(dim=-2).sum(dim=-1)
        union = ((pred + gt) > 0.5).float()
        u = union.sum(dim=-2).sum(dim=-1)
        iou = i / u

        iou[torch.isinf(iou)] = 0.0
        iou[torch.isnan(iou)] = 1.0

        return iou
    # meta哪里来的？
    def forward(self, images, labels, meta):
        # meta batch8
        # meta: ['{"seq_name": "dogs-scale", "obj_id": 4, "frames": [39, 52, 59], "frame0_id": 39}',
        #       '{"seq_name": "scooter-gray", "obj_id": 2, "frames": [41, 58, 40], "frame0_id": 41}',
        #       '{"seq_name": "tuk-tuk", "obj_id": 3, "frames": [24, 33, 27], "frame0_id": 24}',
        #       '{"seq_name": "dogs-scale", "obj_id": 2, "frames": [35, 75, 9], "frame0_id": 35}',
        #       '{"seq_name": "kid-football", "obj_id": 1, "frames": [56, 42, 46], "frame0_id": 56}',
        #       '{"seq_name": "schoolgirls", "obj_id": 6, "frames": [2, 29, 43], "frame0_id": 2}',
        #       '{"seq_name": "miami-surf", "obj_id": 4, "frames": [11, 2, 59], "frame0_id": 11}',
        #       '{"seq_name": "sheep", "obj_id": 1, "frames": [62, 21, 18], "frame0_id": 62}']
        # load frame
        specs = SampleSpec.from_encoded(meta)

        losses = AverageMeter()
        iter_acc = 0
        n = 0
        # 训练网络初始化 cache_hit: 0~batch_size, 有cache时+1
        cache_hits_l3 = self.l3_initialize(images[0], labels[0], specs)
        cache_hits_l4 = self.l4_initialize(images[0], labels[0], specs)
        # print('cache_hits:', cache_hits)
        # print(len(images))
        for i in range(1, len(images)):
            s = self._forward(images[i].to(self.device))
            # 使用nn.BCELoss()时，在_forward()函数末尾加sigmoid()
            # 使用nn.BCEWithLogitsLoss()等其他损失函数时，sigmoid()在计算accuarcy时加
            y = labels[i].to(self.device).float()
            acc = self.compute_accuracy(s.detach(), y)
            loss = self.compute_loss(s, y)


            loss.backward()

            losses.update(loss.item())
            iter_acc += acc.mean().cpu().numpy()
            n += 1

        stats = dict()
        stats['stats/loss'] = losses.avg
        stats['stats/accuracy'] = iter_acc / n
        stats['stats/fcache_hits_l3'] = cache_hits_l3
        stats['stats/fcache_hits_l4'] = cache_hits_l4

        return stats

    def l3_initialize(self, first_image, first_labels, specs):

        cache_hits = 0
        # print('Augment first image and extract features')
        # Augment first image and extract features
        # L = 'layer4'
        L = self.tmodels_l3[0].discriminator.layer
        N = first_image.shape[0]  # Batch size
        for i in range(N):
            # print('frame_name:', specs[i].seq_name)
            state_dict = None
            if self.tmodel_cache_l3.enable:
                state_dict = self.load_target_model(specs[i], self.tmodel_cache_l3, L)

            have_pretrained = (state_dict is not None)

            if not have_pretrained:
                # 没有预训练，tmodel没有cache
                # A 1->num_aug=15
                im, lb = self.augment(first_image[i].to(self.device), first_labels[i].to(self.device))
                # F
                ft = self.feature_extractor.no_grad_forward(im, output_layers=[L], chunk_size=4)

                self.tmodels_l3[i].initialize(ft, lb)

                if self.tmodel_cache_l3.enable and not self.tmodel_cache_l3.read_only:
                    self.save_target_model(specs[i], self.tmodel_cache_l3, L, self.tmodels_l3[i].get_state_dict())

            else:
                if self.ft_channels is None:
                    self.ft_channels = self.feature_extractor.get_out_channels()[L]
                self.tmodels_l3[i].initialize_pretrained(state_dict)
                cache_hits += 1

        return cache_hits

    def l4_initialize(self, first_image, first_labels, specs):

        cache_hits = 0
        # print('Augment first image and extract features')
        # Augment first image and extract features
        # L = 'layer4'
        L = self.tmodels_l4[0].discriminator.layer
        N = first_image.shape[0]  # Batch size
        for i in range(N):
            # print('frame_name:', specs[i].seq_name)
            state_dict = None
            if self.tmodel_cache_l4.enable:
                state_dict = self.load_target_model(specs[i], self.tmodel_cache_l4, L)

            have_pretrained = (state_dict is not None)

            if not have_pretrained:
                # 没有预训练，tmodel没有cache
                # A 1->num_aug=15
                im, lb = self.augment(first_image[i].to(self.device), first_labels[i].to(self.device))
                # F
                ft = self.feature_extractor.no_grad_forward(im, output_layers=[L], chunk_size=4)

                self.tmodels_l4[i].initialize(ft, lb)

                if self.tmodel_cache_l4.enable and not self.tmodel_cache_l4.read_only:
                    self.save_target_model(specs[i], self.tmodel_cache_l4, L, self.tmodels_l4[i].get_state_dict())

            else:
                if self.ft_channels is None:
                    self.ft_channels = self.feature_extractor.get_out_channels()[L]
                self.tmodels_l4[i].initialize_pretrained(state_dict)
                cache_hits += 1

        return cache_hits

    def _forward(self, image):
        # print("image:", torch.max(image), torch.min(image))
        batch_size = image.shape[0]
        # F
        features = self.feature_extractor(image)
        scores_l3, scores_l4 = [], []
        ft_l3 = features[self.tmodels_l3[0].discriminator.layer]
        ft_l4 = features[self.tmodels_l4[0].discriminator.layer]
        # Target model是如何更新的?
        for i, tmd_l3, tmd_l4 in zip(range(batch_size), self.tmodels_l3, self.tmodels_l4):
            x_l3 = ft_l3[i, None]
            x_l4 = ft_l4[i, None]
            # print(x_l4[0, 0, :, :].size())
            # F提取的特征存入D
            s_l3 = tmd_l3.classify(x_l3)
            scores_l3.append(s_l3)
            s_l4 = tmd_l4.classify(x_l4)
            scores_l4.append(s_l4)
        scores_L3 = torch.cat(scores_l3, dim=0)
        scores_L4 = torch.cat(scores_l4, dim=0)
        # print(scores_L3.size(), scores_L4.size(), image.shape)
        # scores_l3 shape:[8, 1, 60, 107], scores_l4 shape:[8, 1, 30, 54]
        # S:正常的Encoder-Decoder
        y = self.refiner(scores_L3, scores_L4, features, image.shape)
        y = interpolate(y, image.shape[-2:])

        return torch.sigmoid(y)
    # tmodel_cache存储格式（frame.id+obj_id+layer_id)
    def tmodel_filename(self, spec, cache,  layer_name):
        return cache.path / spec.seq_name / ("%05d.%d.%s.pth" % (spec.frame0_id, spec.obj_id, layer_name))

    def load_target_model(self, spec, cache, layer_name):
        fname = self.tmodel_filename(spec, cache, layer_name)
        # print('load_target_model fname:', fname)
        try:
            state_dict = torch.load(fname, map_location=self.device) if fname.exists() else None
        except Exception as e:
            print("Could not read %s: %s" % (fname, e))
            state_dict = None
        return state_dict

    def save_target_model(self, spec: SampleSpec, cache, layer_name, state_dict):
        fname = self.tmodel_filename(spec, cache, layer_name)
        #('fname:', fname)
        fname.parent.mkdir(exist_ok=True, parents=True)
        torch.save(state_dict, fname)
