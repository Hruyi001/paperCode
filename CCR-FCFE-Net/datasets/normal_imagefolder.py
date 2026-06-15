from torchvision import datasets
from torchvision.datasets.folder import DatasetFolder, default_loader

from .omnidata_normals import NormalCache


class NormalImageFolder(datasets.ImageFolder):
    extensions = datasets.folder.IMG_EXTENSIONS

    def __init__(self, root, transform=None, target_transform=None, normal_dir='', data_root=None,
                 omnidata_repo='/root/code/omnidata_models', omnidata_root='/root/code/omnidata_models/pretrained_models',
                 auto_generate=True, use_normals=True):
        DatasetFolder.__init__(
            self,
            root,
            loader=default_loader,
            extensions=self.extensions,
            transform=transform,
            target_transform=target_transform,
            allow_empty=True,
        )
        self.imgs = self.samples
        self.use_normals = use_normals
        self.normal_cache = NormalCache(
            normal_dir=normal_dir,
            data_root=data_root or root,
            omnidata_repo=omnidata_repo,
            omnidata_root=omnidata_root,
            auto_generate=auto_generate,
        ) if use_normals else None

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.use_normals:
            if self.transform is not None:
                sample = self.transform(sample)
            return sample, target
        normal = self.normal_cache.get(path)
        if self.transform is not None:
            sample = self.transform(sample)
            normal = self.transform(normal)
        return (sample, normal), target
