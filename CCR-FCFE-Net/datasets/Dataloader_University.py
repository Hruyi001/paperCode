import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as tv_transforms
from torchvision.transforms import functional as TF

from .omnidata_normals import NormalCache
from .random_erasing import RandomErasing


class Dataloader_University(Dataset):
    def __init__(self, root, transforms, names=['satellite', 'street', 'drone', 'google'], use_normals=False,
                 normal_dir='', omnidata_repo='/root/code/omnidata_models',
                 omnidata_root='/root/code/omnidata_models/pretrained_models', auto_generate_normals=True):
        super(Dataloader_University).__init__()
        transform_config = transforms
        self.transforms_drone_street = transform_config['train']
        self.transforms_satellite = transform_config['satellite']
        self.root = root
        self.names = names
        self.use_normals = use_normals
        self.h = transform_config.get('h')
        self.w = transform_config.get('w')
        self.pad = transform_config.get('pad', 0)
        self.erasing_p = transform_config.get('erasing_p', 0)
        self.color_jitter = transform_config.get('color_jitter', False)
        self.normal_cache = NormalCache(
            normal_dir=normal_dir,
            data_root=root,
            omnidata_repo=omnidata_repo,
            omnidata_root=omnidata_root,
            auto_generate=auto_generate_normals,
        ) if use_normals else None
        self.rgb_normalize = tv_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.normal_normalize = tv_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.random_erasing = RandomErasing(probability=self.erasing_p, mean=[0.0, 0.0, 0.0]) if self.erasing_p > 0 else None
        self.rgb_jitter = tv_transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)
        dict_path = {}
        for name in names:
            dict_ = {}
            for cls_name in os.listdir(os.path.join(root, name)):
                img_list = os.listdir(os.path.join(root, name, cls_name))
                img_path_list = [os.path.join(root, name, cls_name, img) for img in img_list]
                dict_[cls_name] = img_path_list
            dict_path[name] = dict_
            # dict_path[name+"/"+cls_name] = img_path_list

        cls_names = os.listdir(os.path.join(root, names[0]))
        cls_names.sort()
        map_dict = {i: cls_names[i] for i in range(len(cls_names))}

        self.cls_names = cls_names
        self.map_dict = map_dict
        self.dict_path = dict_path
        self.index_cls_nums = 2

    def sample_from_cls(self, name, cls_num):
        img_path = self.dict_path[name][cls_num]
        img_path = np.random.choice(img_path, 1)[0]
        img = Image.open(img_path).convert('RGB')
        if self.use_normals:
            return img, self.normal_cache.get(img_path)
        return img

    def _paired_transform(self, rgb, normal, satellite=False):
        if self.color_jitter:
            rgb = self.rgb_jitter(rgb)
        rgb = TF.resize(rgb, (self.h, self.w), interpolation=3)
        normal = TF.resize(normal, (self.h, self.w), interpolation=3)
        if self.pad > 0:
            rgb = TF.pad(rgb, self.pad, padding_mode='edge')
            normal = TF.pad(normal, self.pad, padding_mode='edge')
        if satellite:
            angle = random.uniform(-90, 90)
            rgb = TF.affine(rgb, angle=angle, translate=[0, 0], scale=1.0, shear=[0.0, 0.0], interpolation=3)
            normal = TF.affine(normal, angle=angle, translate=[0, 0], scale=1.0, shear=[0.0, 0.0], interpolation=3)
        i, j, h, w = tv_transforms.RandomCrop.get_params(rgb, output_size=(self.h, self.w))
        rgb = TF.crop(rgb, i, j, h, w)
        normal = TF.crop(normal, i, j, h, w)
        if random.random() < 0.5:
            rgb = TF.hflip(rgb)
            normal = TF.hflip(normal)
        rgb = self.rgb_normalize(TF.to_tensor(rgb))
        normal = self.normal_normalize(TF.to_tensor(normal))
        if self.random_erasing is not None:
            rgb = self.random_erasing(rgb)
        return rgb, normal

    def __getitem__(self, index):
        cls_nums = self.map_dict[index]
        if self.use_normals:
            img, normal = self.sample_from_cls("satellite", cls_nums)
            img_s = self._paired_transform(img, normal, satellite=True)

            img, normal = self.sample_from_cls("street", cls_nums)
            img_st = self._paired_transform(img, normal)

            img, normal = self.sample_from_cls("drone", cls_nums)
            img_d = self._paired_transform(img, normal)
            return img_s, img_st, img_d, index

        img = self.sample_from_cls("satellite", cls_nums)
        img_s = self.transforms_satellite(img)

        img = self.sample_from_cls("street", cls_nums)
        img_st = self.transforms_drone_street(img)

        img = self.sample_from_cls("drone", cls_nums)
        img_d = self.transforms_drone_street(img)
        return img_s, img_st, img_d, index

    def __len__(self):
        return len(self.cls_names)


class Sampler_University(object):
    r"""Base class for all Samplers.
    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices of dataset elements, and a :meth:`__len__` method
    that returns the length of the returned iterators.
    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`~torch.utils.data.DataLoader`, but is expected in any
              calculation involving the length of a :class:`~torch.utils.data.DataLoader`.
    """

    def __init__(self, data_source, batchsize=8, sample_num=4, triplet_loss=0):
        self.data_len = len(data_source)
        self.batchsize = batchsize
        self.sample_num = sample_num
        self.triplet_loss = triplet_loss

    def __iter__(self):
        list = np.arange(0, self.data_len)
        nums = np.repeat(list, self.sample_num, axis=0)
        np.random.shuffle(nums)
        # print(nums)
        return iter(nums)

    def __len__(self):
        return len(self.data_source)


def _stack_view(items):
    if isinstance(items[0], (tuple, list)):
        rgb, normal = zip(*items)
        return torch.stack(rgb, dim=0), torch.stack(normal, dim=0)
    return torch.stack(items, dim=0)


def train_collate_fn(batch):
    img_s, img_st, img_d, ids = zip(*batch)
    ids = torch.tensor(ids, dtype=torch.int64)
    return [_stack_view(img_s), ids], [_stack_view(img_st), ids], [_stack_view(img_d), ids]
