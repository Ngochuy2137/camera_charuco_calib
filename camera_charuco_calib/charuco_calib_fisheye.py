#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cv2.fisheye calibration helper (separate from pinhole / calibrate3d)."""

from typing import Optional, Tuple, Any, List

import cv2
import numpy as np


def build_fisheye_calib_flags(fisheye_extra: int = 0) -> int:
    fe = cv2.fisheye
    base = int(fe.CALIB_RECOMPUTE_EXTRINSIC) | int(fe.CALIB_CHECK_COND) | int(fe.CALIB_FIX_SKEW)
    return int((base | (int(fisheye_extra) & 0xFFFFFFFF)) & 0xFFFFFFFF)


def calibrate_fisheye_from_accumulator(
    acc: Any,
    criteria: Optional[Tuple],
    fisheye_flags_extra: int = 0,
) -> Tuple[float, np.ndarray, np.ndarray, List, List]:
    n_obj = len(acc.all_obj_points)
    n_img = len(acc.all_img_points)
    if n_obj < 5 or n_img < 5 or n_obj != n_img:
        raise RuntimeError(
            'Fisheye needs 3D/2D point lists (at least 5 views). '
            f'Got object point sets: {n_obj}, image point sets: {n_img}. '
            'Update calibrate_camera_node (synthetic ChArUco 3D grid from --width/--height/--square_length), '
            'and ensure ChArUco corner ids are in range. Or use --projection-model pinhole.'
        )

    flags = build_fisheye_calib_flags(fisheye_flags_extra)
    K = np.zeros((3, 3), dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)

    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        acc.all_obj_points,
        acc.all_img_points,
        acc.image_size,
        K,
        D,
        None,
        None,
        flags,
        criteria,
    )
    return rms, K, D, rvecs, tvecs
