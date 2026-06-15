import os
import sys
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms


class NormalCache:
    def __init__(self, normal_dir, data_root=None, omnidata_repo='/root/code/omnidata_models',
                 omnidata_root='/root/code/omnidata_models/pretrained_models', auto_generate=True, device='auto'):
        if not normal_dir:
            normal_dir = os.path.join(data_root or '.', 'omnidata_normals')
        self.normal_dir = os.path.abspath(normal_dir)
        self.data_root = os.path.abspath(data_root) if data_root else None
        self.omnidata_repo = omnidata_repo
        self.omnidata_root = omnidata_root
        self.auto_generate = auto_generate
        self.device = device
        self._model = None
        self._transform = None
        self._actual_device = None

    def normal_path(self, rgb_path):
        rgb_path = os.path.abspath(rgb_path)
        if self.data_root:
            rel_path = os.path.relpath(rgb_path, self.data_root)
        else:
            rel_path = os.path.basename(rgb_path)
        rel = Path(rel_path)
        return str(Path(self.normal_dir) / rel.parent / f'{rel.stem}_normal.png')

    def get(self, rgb_path):
        normal_path = self.normal_path(rgb_path)
        if not os.path.exists(normal_path):
            if not self.auto_generate:
                raise FileNotFoundError(normal_path)
            self.generate(rgb_path, normal_path)
        try:
            with Image.open(normal_path) as image:
                return image.convert('RGB')
        except (OSError, UnidentifiedImageError):
            if not self.auto_generate:
                raise
            try:
                os.remove(normal_path)
            except FileNotFoundError:
                pass
            self.generate(rgb_path, normal_path)
            with Image.open(normal_path) as image:
                return image.convert('RGB')

    def generate(self, rgb_path, normal_path):
        os.makedirs(os.path.dirname(normal_path), exist_ok=True)
        tmp_path = f'{normal_path}.tmp.{os.getpid()}'
        model, transform, device = self._load_model()
        image = Image.open(rgb_path).convert('RGB')
        tensor = transform(image)[:3].unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor).clamp(min=0, max=1)
        transforms.ToPILImage()(output[0].cpu()).save(tmp_path, format='PNG')
        os.replace(tmp_path, normal_path)

    def _load_model(self):
        if self._model is not None:
            return self._model, self._transform, self._actual_device
        if self.omnidata_repo not in sys.path:
            sys.path.insert(0, self.omnidata_repo)
        from demo import setup_model, setup_transforms

        if self.device == 'auto':
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(self.device)
        root_dir = self.omnidata_root if self.omnidata_root else None
        model, image_size, actual_device = setup_model('normal', root_dir, device)
        transform, _ = setup_transforms('normal', image_size)
        self._model = model
        self._transform = transform
        self._actual_device = actual_device
        return self._model, self._transform, self._actual_device
