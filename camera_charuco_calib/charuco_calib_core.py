#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ChArUco board helpers, frame accumulation, OpenCV calibration, and result writers."""

import json
import os
from collections import OrderedDict
from datetime import datetime
from typing import Optional, Tuple

import cv2
import numpy as np

ARUCO_DICTS = {
    'DICT_4X4_50': cv2.aruco.DICT_4X4_50,
    'DICT_4X4_100': cv2.aruco.DICT_4X4_100,
    'DICT_4X4_250': cv2.aruco.DICT_4X4_250,
    'DICT_4X4_1000': cv2.aruco.DICT_4X4_1000,
    'DICT_5X5_50': cv2.aruco.DICT_5X5_50,
    'DICT_5X5_100': cv2.aruco.DICT_5X5_100,
    'DICT_5X5_250': cv2.aruco.DICT_5X5_250,
    'DICT_5X5_1000': cv2.aruco.DICT_5X5_1000,
    'DICT_6X6_50': cv2.aruco.DICT_6X6_50,
    'DICT_6X6_100': cv2.aruco.DICT_6X6_100,
    'DICT_6X6_250': cv2.aruco.DICT_6X6_250,
    'DICT_6X6_1000': cv2.aruco.DICT_6X6_1000,
    'DICT_7X7_50': cv2.aruco.DICT_7X7_50,
    'DICT_7X7_100': cv2.aruco.DICT_7X7_100,
    'DICT_7X7_250': cv2.aruco.DICT_7X7_250,
    'DICT_7X7_1000': cv2.aruco.DICT_7X7_1000,
    'DICT_ARUCO_ORIGINAL': cv2.aruco.DICT_ARUCO_ORIGINAL,
}

for name in ['DICT_APRILTAG_16h5', 'DICT_APRILTAG_25h9', 'DICT_APRILTAG_36h10', 'DICT_APRILTAG_36h11']:
    if hasattr(cv2.aruco, name):
        ARUCO_DICTS[name] = getattr(cv2.aruco, name)


def get_aruco_dict(dict_name: str):
    if dict_name not in ARUCO_DICTS:
        raise ValueError(f'Unknown dictionary: {dict_name}')
    dict_id = ARUCO_DICTS[dict_name]

    if hasattr(cv2.aruco, 'getPredefinedDictionary'):
        return cv2.aruco.getPredefinedDictionary(dict_id)
    return cv2.aruco.Dictionary_get(dict_id)


def create_charuco_board(
    squares_x: int,
    squares_y: int,
    square_len: float,
    marker_len: float,
    aruco_dict,
    prefer_legacy: bool = True,
):
    """
    prefer_legacy=True (default, pinhole path): prefer CharucoBoard_create when available — on
    some Jetson/OpenCV 4.6 builds the new CharucoBoard + interpolateCornersCharuco can segfault.

    prefer_legacy=False (fisheye path): prefer CharucoBoard (new) first so `matchImagePoints` /
    `getChessboardCorners` are available to build 3D/2D points for `cv2.fisheye.calibrate`.
    Returns (board, allow_charuco_detector).
    """
    if not prefer_legacy and hasattr(cv2.aruco, 'CharucoBoard'):
        board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y),
            square_len,
            marker_len,
            aruco_dict
        )
        return board, True

    if prefer_legacy and hasattr(cv2.aruco, 'CharucoBoard_create'):
        board = cv2.aruco.CharucoBoard_create(
            squares_x,
            squares_y,
            square_len,
            marker_len,
            aruco_dict
        )
        return board, False

    if hasattr(cv2.aruco, 'CharucoBoard'):
        board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y),
            square_len,
            marker_len,
            aruco_dict
        )
        return board, True

    if hasattr(cv2.aruco, 'CharucoBoard_create'):
        board = cv2.aruco.CharucoBoard_create(
            squares_x,
            squares_y,
            square_len,
            marker_len,
            aruco_dict
        )
        return board, False

    raise RuntimeError('Your cv2.aruco does not support CharucoBoard.')


def create_detector(board, allow_charuco_detector: bool):
    if not allow_charuco_detector:
        return None
    if hasattr(cv2.aruco, 'CharucoDetector'):
        return cv2.aruco.CharucoDetector(board)
    return None


def detect_charuco_compatible(gray, board, aruco_dict, detector=None):
    """
    Return:
        charuco_corners, charuco_ids, marker_corners, marker_ids
    """
    if detector is not None:
        corners, ids, marker_corners, marker_ids = detector.detectBoard(gray)
        return corners, ids, marker_corners, marker_ids

    if hasattr(cv2.aruco, 'DetectorParameters_create'):
        params = cv2.aruco.DetectorParameters_create()
    else:
        params = cv2.aruco.DetectorParameters()

    marker_corners, marker_ids, _rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

    if marker_ids is None or len(marker_ids) == 0:
        return None, None, marker_corners, marker_ids

    retval = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray, board)

    if len(retval) == 3:
        maybe_count, charuco_corners, charuco_ids = retval
        if isinstance(maybe_count, (int, float)) and maybe_count <= 0:
            return None, None, marker_corners, marker_ids
    elif len(retval) == 2:
        charuco_corners, charuco_ids = retval
    else:
        raise RuntimeError('Unexpected return from interpolateCornersCharuco')

    return charuco_corners, charuco_ids, marker_corners, marker_ids


def _synthetic_charuco_chessboard_xyz(squares_x: int, squares_y: int, square_len: float) -> np.ndarray:
    """
    Same 3D layout as OpenCV CharucoBoard::getChessboardCorners (verified against cv2.aruco.CharucoBoard).
    Inner corners: (squares_x-1)*(squares_y-1), Z=0, X=(x+1)*square_len, Y=(y+1)*square_len.
    """
    pts = []
    for y in range(0, int(squares_y) - 1):
        for x in range(0, int(squares_x) - 1):
            pts.append(
                [float((x + 1) * square_len), float((y + 1) * square_len), 0.0]
            )
    if not pts:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


def _pair_charuco_obj_img_from_xyz_rows(
    all_xyz: np.ndarray,
    charuco_corners,
    charuco_ids,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Map each charuco id to a 3D row in all_xyz and pair with the corresponding image corner."""
    if charuco_ids is None or charuco_corners is None:
        return None, None
    all_xyz = np.asarray(all_xyz, dtype=np.float32)
    if all_xyz.size == 0:
        return None, None
    if all_xyz.ndim != 2 or all_xyz.shape[1] != 3:
        return None, None
    n_all = int(all_xyz.shape[0])
    ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
    cc = np.asarray(charuco_corners, dtype=np.float32)
    if cc.ndim == 2:
        cc = cc[:, np.newaxis, :]
    obj_list = []
    img_list = []
    n = min(int(ids.size), int(cc.shape[0]))
    for i in range(n):
        cid = int(ids[i])
        if cid < 0 or cid >= n_all:
            continue
        obj_list.append(all_xyz[cid].reshape(1, 1, 3).astype(np.float32))
        img_list.append(cc[i].reshape(1, 1, 2).astype(np.float32))
    if len(obj_list) < 4:
        return None, None
    return np.vstack(obj_list), np.vstack(img_list)


def _charuco_object_image_from_synthetic_grid(
    squares_x: int,
    squares_y: int,
    square_len: float,
    charuco_corners,
    charuco_ids,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """When the board object has no getChessboardCorners in Python (legacy API)."""
    all_xyz = _synthetic_charuco_chessboard_xyz(squares_x, squares_y, square_len)
    return _pair_charuco_obj_img_from_xyz_rows(all_xyz, charuco_corners, charuco_ids)


def _charuco_object_image_from_chessboard_3d(board, charuco_corners, charuco_ids):
    """
    When `matchImagePoints` is missing (legacy CharucoBoard) or returns nothing, build
    3D/2D pairs from getChessboardCorners()[id] and image corners (OpenCV ChArUco id indexing).
    """
    if not hasattr(board, 'getChessboardCorners') or charuco_ids is None or charuco_corners is None:
        return None, None
    all_xyz = board.getChessboardCorners()
    if all_xyz is None:
        return None, None
    all_xyz = np.asarray(all_xyz, dtype=np.float32)
    if all_xyz.size == 0:
        return None, None
    if all_xyz.ndim == 3 and all_xyz.shape[1] == 1 and all_xyz.shape[2] == 3:
        all_xyz = all_xyz.reshape(-1, 3)
    elif all_xyz.ndim != 2 or all_xyz.shape[1] != 3:
        return None, None
    return _pair_charuco_obj_img_from_xyz_rows(all_xyz, charuco_corners, charuco_ids)


def match_image_points_compatible(board, charuco_corners, charuco_ids):
    if hasattr(board, 'matchImagePoints'):
        try:
            o, i = board.matchImagePoints(charuco_corners, charuco_ids)
        except Exception:
            o, i = None, None
        if o is not None and i is not None and (np.asarray(o).size > 0 and np.asarray(i).size > 0):
            return o, i
    o2, i2 = _charuco_object_image_from_chessboard_3d(board, charuco_corners, charuco_ids)
    if o2 is not None and i2 is not None:
        return o2, i2
    return None, None


def flatten_ids(ids) -> np.ndarray:
    ids = np.asarray(ids)
    return ids.reshape(-1, 1).astype(np.int32)


def ensure_bgr(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def ensure_gray(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


DEBUG_OVERLAY_SUBDIR = 'overlay'
DEBUG_RAW_SUBDIR = 'raw'


def save_debug_image(debug_dir: Optional[str], name: str, gray, marker_corners, marker_ids, charuco_corners, charuco_ids):
    if not debug_dir:
        return None
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if marker_ids is not None and len(marker_ids) > 0:
        cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)

    if charuco_ids is not None and len(charuco_ids) > 0:
        cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)

    out_dir = os.path.join(debug_dir, DEBUG_OVERLAY_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{name}_detected.png')
    cv2.imwrite(out_path, vis)
    return out_path


def save_debug_raw_image(debug_dir: Optional[str], name: str, frame_bgr, gray):
    """Save the original capture (BGR or grayscale) for use with --source file later."""
    if not debug_dir or gray is None:
        return None
    out_dir = os.path.join(debug_dir, DEBUG_RAW_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{name}.png')
    if frame_bgr is not None:
        cv2.imwrite(out_path, frame_bgr)
    else:
        cv2.imwrite(out_path, gray)
    return out_path


def draw_live_overlay(frame_bgr, marker_corners, marker_ids, charuco_corners, charuco_ids, info_text):
    vis = frame_bgr.copy()

    if marker_ids is not None and len(marker_ids) > 0:
        cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)

    if charuco_ids is not None and len(charuco_ids) > 0:
        cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)

    y = 30
    for line in info_text:
        cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        y += 28

    return vis


class CalibrationAccumulator:
    def __init__(
        self,
        board,
        min_corners: int,
        debug_dir: Optional[str],
        save_captures_dir: Optional[str],
        squares_x: Optional[int] = None,
        squares_y: Optional[int] = None,
        square_len: Optional[float] = None,
    ):
        self.board = board
        self.min_corners = min_corners
        self.debug_dir = debug_dir
        self.save_captures_dir = save_captures_dir
        self._sq_x = squares_x
        self._sq_y = squares_y
        self._sq_len = square_len

        self.all_obj_points = []
        self.all_img_points = []
        self.all_charuco_corners = []
        self.all_charuco_ids = []
        self.image_size: Optional[Tuple[int, int]] = None
        self.kept = 0

    def try_add_frame(self, frame_bgr, gray, frame_name, marker_corners, marker_ids, charuco_corners, charuco_ids):
        if gray is None:
            print(f'[SKIP] {frame_name}: gray image is None')
            return False

        current_size = (gray.shape[1], gray.shape[0])
        if self.image_size is None:
            self.image_size = current_size
        elif self.image_size != current_size:
            print(f'[SKIP] {frame_name}: different image size {current_size}, expected {self.image_size}')
            return False

        if charuco_ids is None or charuco_corners is None:
            print(f'[SKIP] {frame_name}: no ChArUco corners')
            save_debug_image(self.debug_dir, frame_name, gray, marker_corners, marker_ids, [], [])
            save_debug_raw_image(self.debug_dir, frame_name, frame_bgr, gray)
            return False

        num_corners = len(charuco_ids)
        if num_corners < self.min_corners:
            print(f'[SKIP] {frame_name}: too few corners ({num_corners})')
            save_debug_image(self.debug_dir, frame_name, gray, marker_corners, marker_ids, charuco_corners, charuco_ids)
            save_debug_raw_image(self.debug_dir, frame_name, frame_bgr, gray)
            return False

        frame_obj_points, frame_img_points = match_image_points_compatible(self.board, charuco_corners, charuco_ids)
        if (frame_obj_points is None or frame_img_points is None) and (
            self._sq_x is not None and self._sq_y is not None and self._sq_len is not None
        ):
            frame_obj_points, frame_img_points = _charuco_object_image_from_synthetic_grid(
                int(self._sq_x),
                int(self._sq_y),
                float(self._sq_len),
                charuco_corners,
                charuco_ids,
            )

        if frame_obj_points is not None and frame_img_points is not None:
            n_obj = int(np.asarray(frame_obj_points).shape[0])
            n_img = int(np.asarray(frame_img_points).shape[0])
            if n_obj < 4 or n_img < 4 or n_obj != n_img:
                print(f'[SKIP] {frame_name}: not enough matched points')
                save_debug_image(self.debug_dir, frame_name, gray, marker_corners, marker_ids, charuco_corners, charuco_ids)
                save_debug_raw_image(self.debug_dir, frame_name, frame_bgr, gray)
                return False

            self.all_obj_points.append(np.asarray(frame_obj_points, dtype=np.float32))
            self.all_img_points.append(np.asarray(frame_img_points, dtype=np.float32))

        self.all_charuco_corners.append(np.asarray(charuco_corners, dtype=np.float32))
        self.all_charuco_ids.append(flatten_ids(charuco_ids))

        self.kept += 1
        save_debug_image(self.debug_dir, frame_name, gray, marker_corners, marker_ids, charuco_corners, charuco_ids)
        save_debug_raw_image(self.debug_dir, frame_name, frame_bgr, gray)

        if self.save_captures_dir is not None:
            os.makedirs(self.save_captures_dir, exist_ok=True)
            raw_path = os.path.join(self.save_captures_dir, f'{frame_name}.png')
            cv2.imwrite(raw_path, frame_bgr)

        print(f'[OK] {frame_name}: kept, corners={num_corners}, total={self.kept}')
        return True


def build_opencv_calib_flags(
    projection_model: str,
    distortion_model: str,
    calib_flags_extra: int = 0,
) -> int:
    if projection_model != 'pinhole':
        raise ValueError(
            f'build_opencv_calib_flags is only for pinhole. Got: {projection_model!r}.'
        )

    flags = int(calib_flags_extra) & 0xFFFFFFFF
    if distortion_model == 'brown_rational':
        rmod = int(getattr(cv2, 'CALIB_RATIONAL_MODEL', 0))
        if rmod == 0:
            raise ValueError('OpenCV has no CALIB_RATIONAL_MODEL; use --distortion-model brown.')
        flags |= rmod
    elif distortion_model == 'brown':
        pass
    else:
        raise ValueError(
            f'Unknown distortion_model: {distortion_model!r}. Use "brown" or "brown_rational".'
        )

    return int(flags)


def _make_term_criteria(
    calib_max_iter: Optional[int],
    calib_epsilon: Optional[float],
) -> Optional[Tuple[int, int, float]]:
    if calib_max_iter is None and calib_epsilon is None:
        return None
    max_iter = 30 if calib_max_iter is None else int(calib_max_iter)
    eps = 1e-5 if calib_epsilon is None else float(calib_epsilon)
    if max_iter < 1:
        raise ValueError('--calib-max-iter must be >= 1 when set')
    if eps <= 0:
        raise ValueError('--calib-epsilon must be > 0 when set')
    return (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, max_iter, eps)


def _calibrate_camera_charuco_with_flags(
    acc: CalibrationAccumulator,
    board,
    flags: int,
    criteria: Optional[Tuple],
):
    """Call cv2.aruco.calibrateCameraCharuco; Python bindings differ across OpenCV 4.x."""
    if not hasattr(cv2.aruco, 'calibrateCameraCharuco'):
        return None
    charuco_fn = cv2.aruco.calibrateCameraCharuco
    a = (
        acc.all_charuco_corners,
        acc.all_charuco_ids,
        board,
        acc.image_size,
        None,
        None,
    )
    try:
        return charuco_fn(*a, None, None, flags, criteria)
    except TypeError:
        pass
    for kwargs in (
        dict(rvecs=None, tvecs=None, flag=flags, criteria=criteria),
        dict(rvecs=None, tvecs=None, flags=flags, criteria=criteria),
    ):
        try:
            return charuco_fn(*a, **kwargs)
        except TypeError:
            continue
    raise TypeError('calibrateCameraCharuco: no matching signature for this OpenCV build')


def calibrate_from_accumulator(
    acc: CalibrationAccumulator,
    board,
    projection_model: str = 'pinhole',
    distortion_model: str = 'brown',
    calib_flags_extra: int = 0,
    calib_max_iter: Optional[int] = None,
    calib_epsilon: Optional[float] = None,
    fisheye_flags_extra: int = 0,
):
    if acc.kept < 5:
        raise RuntimeError('Too few valid images/frames. Try collecting 15-30 from varied viewpoints.')

    if acc.image_size is None:
        raise RuntimeError('No valid image size available.')

    criteria = _make_term_criteria(calib_max_iter, calib_epsilon)

    if projection_model == 'fisheye':
        from camera_charuco_calib.charuco_calib_fisheye import calibrate_fisheye_from_accumulator

        return calibrate_fisheye_from_accumulator(
            acc,
            criteria,
            fisheye_flags_extra=fisheye_flags_extra,
        )

    flags = build_opencv_calib_flags(projection_model, distortion_model, calib_flags_extra)

    if len(acc.all_obj_points) >= 5 and len(acc.all_img_points) >= 5:
        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            acc.all_obj_points,
            acc.all_img_points,
            acc.image_size,
            None,
            None,
            None,
            None,
            flags,
            criteria,
        )
    else:
        out = _calibrate_camera_charuco_with_flags(acc, board, flags, criteria)
        if out is None:
            raise RuntimeError('Neither matchImagePoints nor calibrateCameraCharuco is available.')
        rms, camera_matrix, dist_coeffs, rvecs, tvecs = out

    return rms, camera_matrix, dist_coeffs, rvecs, tvecs


def save_calibration_yaml(
    path,
    image_size,
    args,
    rms,
    camera_matrix,
    dist_coeffs,
    resolved_calib_flags: Optional[int] = None,
    resolved_fisheye_flags: Optional[int] = None,
):
    path = os.path.expanduser(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write('image_width', image_size[0])
    fs.write('image_height', image_size[1])
    fs.write('board_squares_x', args.width)
    fs.write('board_squares_y', args.height)
    fs.write('square_length', args.square_length)
    fs.write('marker_size', args.marker_size)
    fs.write('aruco_dict', args.aruco_dict)
    fs.write('source', args.source)
    if hasattr(args, 'projection_model'):
        fs.write('projection_model', str(args.projection_model))
    if hasattr(args, 'distortion_model'):
        fs.write('distortion_model', str(args.distortion_model))
    if resolved_calib_flags is not None:
        fs.write('calib_flags', int(resolved_calib_flags))
    if resolved_fisheye_flags is not None:
        fs.write('fisheye_flags', int(resolved_fisheye_flags))
    fs.write('rms', float(rms))
    fs.write('camera_matrix', camera_matrix)
    fs.write('dist_coeffs', dist_coeffs)
    fs.release()


def save_calibration_json(
    path: str,
    image_size: Tuple[int, int],
    camera_matrix,
    dist_coeffs,
    rms: float,
    requested_width: Optional[int] = None,
    requested_height: Optional[int] = None,
    requested_fps: int = 30,
    projection_model: Optional[str] = None,
    distortion_model: Optional[str] = None,
    calib_flags: Optional[int] = None,
    fisheye_flags: Optional[int] = None,
):
    img_w, img_h = int(image_size[0]), int(image_size[1])
    rw = int(requested_width) if requested_width is not None else img_w
    rh = int(requested_height) if requested_height is not None else img_h
    dt_string = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    data = OrderedDict([
        ('calibration_time', dt_string),
        ('image_width', img_w),
        ('image_height', img_h),
        ('requested_width', rw),
        ('requested_height', rh),
        ('requested_fps', int(requested_fps)),
        ('camera_matrix', np.asarray(camera_matrix).tolist()),
        ('distortion_coefficients', np.asarray(dist_coeffs).tolist()),
        ('avg_reprojection_error', float(rms)),
    ])
    if projection_model is not None:
        data['projection_model'] = projection_model
    if distortion_model is not None:
        data['distortion_model'] = distortion_model
    if calib_flags is not None:
        data['calib_flags'] = int(calib_flags)
    if fisheye_flags is not None:
        data['fisheye_flags'] = int(fisheye_flags)
    parent = os.path.dirname(os.path.expanduser(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    out_path = os.path.expanduser(path)
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=4)
    return out_path


def save_ros_usb_cam_camera_info_yaml(
    path: str,
    image_size: Tuple[int, int],
    camera_matrix,
    dist_coeffs,
    camera_name: str = 'usb_cam_color',
    distortion_model: str = 'plumb_bob',
    rms: Optional[float] = None,
) -> str:
    """
    Write ROS camera_calibration-style YAML for usb_cam `camera_info_url`
    (sensor_msgs/CameraInfo). Mono pinhole: rectification_matrix = I, projection_matrix from K
    with the fourth column zero (no stereo baseline).
    """
    path = os.path.expanduser(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    w, h = int(image_size[0]), int(image_size[1])
    K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)

    d = np.asarray(dist_coeffs, dtype=np.float64).ravel()
    if distortion_model == 'plumb_bob':
        if d.size < 5:
            d = np.pad(d, (0, 5 - d.size))
    elif distortion_model == 'equidistant':
        if d.size < 4:
            d = np.pad(d, (0, 4 - d.size))
        elif d.size > 4:
            d = d[:4].copy()

    def _fmt_row(vals):
        return '[' + ', '.join(f'{float(x):.6f}' for x in vals) + ']'

    r_data = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    p_data = [
        float(K[0, 0]),
        float(K[0, 1]),
        float(K[0, 2]),
        0.0,
        float(K[1, 0]),
        float(K[1, 1]),
        float(K[1, 2]),
        0.0,
        float(K[2, 0]),
        float(K[2, 1]),
        float(K[2, 2]),
        0.0,
    ]
    k_data = [float(K[i, j]) for i in range(3) for j in range(3)]

    lines = [
        '# ROS camera_calibration YAML for usb_cam camera_info_url (sensor_msgs/CameraInfo).',
        '# Generated by camera_charuco_calib calibrate_camera_node.py.',
        '# Mono: rectification_matrix = I; P from K.',
    ]
    if rms is not None:
        lines.append(f'# avg_reprojection_error: {float(rms):.6f}')
    lines.extend(
        [
            f'image_width: {w}',
            f'image_height: {h}',
            f'camera_name: {camera_name}',
            'camera_matrix:',
            '  rows: 3',
            '  cols: 3',
            f'  data: {_fmt_row(k_data)}',
            f'distortion_model: {distortion_model}',
            'distortion_coefficients:',
            '  rows: 1',
            f'  cols: {int(d.size)}',
            f'  data: {_fmt_row(d)}',
            'rectification_matrix:',
            '  rows: 3',
            '  cols: 3',
            f'  data: {_fmt_row(r_data)}',
            'projection_matrix:',
            '  rows: 3',
            '  cols: 4',
            f'  data: {_fmt_row(p_data)}',
            '',
        ]
    )
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return path
