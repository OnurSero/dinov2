#!/usr/bin/env python3

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.decomposition import PCA

def chw_to_hwc(data: np.ndarray) -> np.ndarray:
    """Converts a Numpy array from CHW to HWC (C = channels, H = height, W = width).

    Args:
        data: A Numpy array width dimensions in the CHW order.
    Returns:
        A Numpy array width dimensions in the HWC order.
    """

    return np.transpose(data, (1, 2, 0))

def resize_image(
    image: np.ndarray,
    size: Tuple[int, int],
    interpolation: Optional[Any] = None,
) -> np.ndarray:
    """Resizes an image.

    Args:
      image: An input image.
      size: The size of the output image (width, height).
      interpolation: An interpolation method (a suitable one is picked if undefined).
    Returns:
      The resized image.
    """

    if interpolation is None:
        interpolation = (
            cv2.INTER_AREA if image.shape[0] >= size[1] else cv2.INTER_LINEAR
        )
    return cv2.resize(image, size, interpolation=interpolation)

def normalize_data(img):
    return (img - img.min()) / (img.max() - img.min())

def tensor_to_array(tensor: torch.Tensor) -> np.ndarray:
    """Converts a tensor into a Numpy array.

    Args:
        tensor: A tensor (may be in the GPU memory).
    Returns:
        A Numpy array.
    """

    return tensor.detach().cpu().numpy()

def vis_pca_feature_map(
    feature_map_chw: np.ndarray,
    image_height: int,
    image_width: int,
):
    pca = PCA(n_components=3)

    # PCA visualization.
    feature_map_chw_up = F.interpolate(
        feature_map_chw.unsqueeze(0),
        size=(image_height, image_width),
        mode="nearest",
    )[0]
    feature_map_hwc_np = chw_to_hwc(tensor_to_array(feature_map_chw_up))

    vis_pca_components = min(pca.n_components, 3)  # 6)
    map_width = feature_map_hwc_np.shape[1]
    map_height = feature_map_hwc_np.shape[0]
    reshaped_features = feature_map_hwc_np.reshape(map_width * map_height, -1)
    pca.fit(reshaped_features)
    query_pca_transform = pca.transform(reshaped_features)
    query_pca_feature_map = query_pca_transform.reshape((map_height, map_width, -1))

    vis_pca_features = None
    for i in range(vis_pca_components // 3):
        vis_pca_features_each = (
            255
            * normalize_data(
                query_pca_feature_map[:, :, i * 3 : (i + 1) * 3]
            )
        ).astype(np.uint8)
        if i == 0:
            vis_pca_features = vis_pca_features_each

    # Make sure the feature map is of the expected size.
    vis_pca_features = resize_image(
        image=vis_pca_features,
        size=(image_width, image_height),
        interpolation=cv2.INTER_NEAREST,
    )

    return vis_pca_features