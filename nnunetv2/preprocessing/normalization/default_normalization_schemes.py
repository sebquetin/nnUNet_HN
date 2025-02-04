from abc import ABC, abstractmethod
from typing import Type

import numpy as np
from numpy import number
import os
import inspect
import json
from pathlib import Path
from nnunetv2.training.nnUNetTrainer.state import ExperimentState


def get_stats_dir():
    local_file_path = os.path.abspath( inspect.getfile(inspect.currentframe()))
    repo_src_path = Path(local_file_path).parents[4]
    
    stats_file_path = os.path.join(repo_src_path, "src", "utils", "statistics.json")
    assert os.path.exists(stats_file_path), f"Statistics file {stats_file_path} does not exist/has not been created yet with get_statistics.py script"
    with open(stats_file_path, "r") as f:
        stats = json.load(f)
    return stats


class ImageNormalization(ABC):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = None

    def __init__(self, use_mask_for_norm: bool = None, intensityproperties: dict = None,
                 target_dtype: Type[number] = np.float32):
        assert use_mask_for_norm is None or isinstance(use_mask_for_norm, bool)
        self.use_mask_for_norm = use_mask_for_norm
        assert isinstance(intensityproperties, dict)
        self.intensityproperties = intensityproperties
        self.target_dtype = target_dtype

    @abstractmethod
    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """
        Image and seg must have the same shape. Seg is not always used
        """
        pass


class ZScoreNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = True

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """
        here seg is used to store the zero valued region. The value for that region in the segmentation is -1 by
        default.
        """
        print("Normal nnunet std for MRI")

        image = image.astype(self.target_dtype, copy=False)
        if self.use_mask_for_norm is not None and self.use_mask_for_norm:
            # negative values in the segmentation encode the 'outside' region (think zero values around the brain as
            # in BraTS). We want to run the normalization only in the brain region, so we need to mask the image.
            # The default nnU-net sets use_mask_for_norm to True if cropping to the nonzero region substantially
            # reduced the image size.
            mask = seg >= 0
            mean = image[mask].mean()
            std = image[mask].std()
            image[mask] = (image[mask] - mean) / (max(std, 1e-8))
        else:
            mean = image.mean()
            std = image.std()
            image -= mean
            image /= (max(std, 1e-8))
        return image
    
class MriSoftNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = True

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """
        here seg is used to store the zero valued region. The value for that region in the segmentation is -1 by
        default.
        """
        image = image.astype(self.target_dtype, copy=False)
        ct_only = ExperimentState.ct_only
        if ct_only:
            return 0*image
        
        stats = get_stats_dir()
        if ExperimentState.reg_rigid_only:
            print("MriSoftNorm reg rigid only")
            mean = float(stats["MRI_soft_tissue_regrigid"]["mean"])
            std = float(stats["MRI_soft_tissue_regrigid"]["std"])
        else:
            print("MriSoftNorm reg both")
            mean = float(stats["MRI_soft_tissue_regboth"]["mean"])
            std = float(stats["MRI_soft_tissue_regboth"]["std"])
     
        image -= mean
        image /= (max(std, 1e-8))

        return image


class CTNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert self.intensityproperties is not None, "CTNormalization requires intensity properties"
        print("Normal nnunet std for CT")
        mean_intensity = self.intensityproperties['mean']
        std_intensity = self.intensityproperties['std']
        lower_bound = self.intensityproperties['percentile_00_5']
        upper_bound = self.intensityproperties['percentile_99_5']

        image = image.astype(self.target_dtype, copy=False)
        np.clip(image, lower_bound, upper_bound, out=image)
        image -= mean_intensity
        image /= max(std_intensity, 1e-8)
        return image


class CTBoneNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert self.intensityproperties is not None, "CTNormalization requires intensity properties"
        
        image = image.astype(self.target_dtype, copy=False)
        mri_only = ExperimentState.mri_only
        if mri_only:
            return 0*image

        print("in CTBoneNorm")
        stats = get_stats_dir()
        mean = float(stats["bone_tissue_HUs"]["mean"])
        std = float(stats["bone_tissue_HUs"]["std"])
        image -= mean
        image /= max(std, 1e-8)
        return image


class CTSoftNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert self.intensityproperties is not None, "CTNormalization requires intensity properties"

        image = image.astype(self.target_dtype, copy=False)
        mri_only = ExperimentState.mri_only
        if mri_only:
            return 0*image
        print("in CTSoftNorm")
        stats = get_stats_dir()
        mean = float(stats["soft_tissue_HUs"]["mean"])
        std = float(stats["soft_tissue_HUs"]["std"])
        image -= mean
        image /= max(std, 1e-8)
        return image


class NoNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        return image.astype(self.target_dtype, copy=False)


class RescaleTo01Normalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        image = image.astype(self.target_dtype, copy=False)
        image -= image.min()
        image /= np.clip(image.max(), a_min=1e-8, a_max=None)
        return image


class RGBTo01Normalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert image.min() >= 0, "RGB images are uint 8, for whatever reason I found pixel values smaller than 0. " \
                                 "Your images do not seem to be RGB images"
        assert image.max() <= 255, "RGB images are uint 8, for whatever reason I found pixel values greater than 255" \
                                   ". Your images do not seem to be RGB images"
        image = image.astype(self.target_dtype, copy=False)
        image /= 255.
        return image

