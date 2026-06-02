#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Load calibration frames from image files on disk."""

import glob
import os

import cv2

from camera_charuco_calib.charuco_calib_core import CalibrationAccumulator, detect_charuco_compatible, ensure_gray


def run_file_source(args, board, aruco_dict, detector, acc: CalibrationAccumulator):
    image_paths = []
    for pattern in args.images:
        image_paths.extend(sorted(glob.glob(pattern)))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        raise RuntimeError('No images found for file source.')

    for path in image_paths:
        frame_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            print(f'[SKIP] Cannot read image: {path}')
            continue

        gray = ensure_gray(frame_bgr)

        try:
            charuco_corners, charuco_ids, marker_corners, marker_ids = detect_charuco_compatible(
                gray, board, aruco_dict, detector
            )
        except Exception as e:
            print(f'[SKIP] Detection error on {path}: {e}')
            continue

        frame_name = os.path.splitext(os.path.basename(path))[0]
        acc.try_add_frame(frame_bgr, gray, frame_name, marker_corners, marker_ids, charuco_corners, charuco_ids)
