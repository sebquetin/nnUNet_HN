import random
from typing import List

import torch

from batchgenerators.transforms.abstract_transforms import AbstractTransform
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform


class MaskTransform(AbstractTransform):
    def __init__(self, apply_to_channels: List[int], mask_idx_in_seg: int = 0, set_outside_to: int = 0,
                 data_key: str = "data", seg_key: str = "seg"):
        """
        Sets everything outside the mask to 0. CAREFUL! outside is defined as < 0, not =0 (in the Mask)!!!
        """
        self.apply_to_channels = apply_to_channels
        self.seg_key = seg_key
        self.data_key = data_key
        self.set_outside_to = set_outside_to
        self.mask_idx_in_seg = mask_idx_in_seg

    def __call__(self, **data_dict):
        mask = data_dict[self.seg_key][:, self.mask_idx_in_seg] < 0
        for c in self.apply_to_channels:
            data_dict[self.data_key][:, c][mask] = self.set_outside_to
        return data_dict

### Version when nnunetv2 was using its own augmentations and batchgeneratorsv1
# The transform sees the the data_dict input which is (batch_size, channels, x, y, z).
class DropoutTransform(AbstractTransform):
    def __init__(self, apply_to_channels: List[int], set_to: float = 0.0, data_key: str = "data"):
        """
        Randomly sets channel from apply_to_channels to 0, with equal probability.
        """
        self.apply_to_channels = list(apply_to_channels)
        self.apply_to_channels.append(-1)
        self.data_key = data_key
        self.set_to = set_to

    def __call__(self, **data_dict):
        random.shuffle(self.apply_to_channels)
        for b in range(data_dict[self.data_key].shape[0]):
            c = random.choice(self.apply_to_channels)
            if c == -1:
                continue
            data_dict[self.data_key][b, c] = self.set_to
        return data_dict
    
### Newest version when nnunetv2 is using batchgeneratorsv2
# The transform sees the the tensor input which is ( channels, x, y, z).
class DropoutTransformV2(BasicTransform):
    def __init__(self, apply_to_channels: List[int], set_to: float = 0.0, data_key: str = "data"):
        """
        Randomly sets channel from apply_to_channels to 0, with equal probability.
        """
        self.apply_to_channels = list(apply_to_channels)
        self.apply_to_channels.append(-1)
        self.data_key = data_key
        self.set_to = set_to

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        random.shuffle(self.apply_to_channels)
        c = random.choice(self.apply_to_channels)
        if c == -1:
            return img
        else:
            # Comma is super important here since c can take either 0 or (1,2) as a value
            # and the tensor should zero-out only the first dimension which in our use case
            # is MRI, CT1, CT2.
            img[c,] = self.set_to
            return img
        
    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        return segmentation
    