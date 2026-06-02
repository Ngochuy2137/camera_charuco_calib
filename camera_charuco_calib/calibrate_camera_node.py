#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChArUco camera calibration entry point.

Delegates frame collection to calib_sources.* modules and calibration I/O to charuco_calib_core.
"""

import argparse
import os
import re

import cv2

_NUMERIC_FLAG = re.compile(r'^[+-]?(?:0[xX][0-9a-fA-F]+|[0-9]+)$')


def parse_calib_extra_flags(value: str) -> int:
    s = str(value).strip()
    if not s:
        return 0
    if ',' in s:
        out = 0
        for part in s.split(','):
            out |= _parse_one_calib_flag_token(part)
        return int(out) & 0xFFFFFFFF
    if _NUMERIC_FLAG.match(s):
        return int(s, 0) & 0xFFFFFFFF
    return int(_parse_one_calib_flag_token(s)) & 0xFFFFFFFF


def _parse_one_calib_flag_token(token: str) -> int:
    t = token.strip()
    if not t:
        return 0
    if _NUMERIC_FLAG.match(t):
        return int(t, 0)
    name = t
    if not name.startswith('CALIB_') and hasattr(cv2, f'CALIB_{name}'):
        name = f'CALIB_{name}'
    if not hasattr(cv2, name):
        calib_names = sorted(n for n in dir(cv2) if n.startswith('CALIB_'))
        hint = ', '.join(calib_names[:12]) + ('...' if len(calib_names) > 12 else '')
        raise argparse.ArgumentTypeError(
            f'Unknown OpenCV calib flag {token!r}. Use a decimal, 0x..., or a name like '
            f'CALIB_RATIONAL_MODEL. Example names: {hint}'
        )
    v = getattr(cv2, name)
    if not isinstance(v, int):
        raise argparse.ArgumentTypeError(f'{name} is not an integer; got {type(v).__name__}')
    return int(v)


def parse_fisheye_extra_flags(value: str) -> int:
    s = str(value).strip()
    if not s:
        return 0
    if ',' in s:
        out = 0
        for part in s.split(','):
            out |= _parse_one_fisheye_flag_token(part)
        return int(out) & 0xFFFFFFFF
    if _NUMERIC_FLAG.match(s):
        return int(s, 0) & 0xFFFFFFFF
    return int(_parse_one_fisheye_flag_token(s)) & 0xFFFFFFFF


def _parse_one_fisheye_flag_token(token: str) -> int:
    t = token.strip()
    if not t:
        return 0
    if _NUMERIC_FLAG.match(t):
        return int(t, 0)
    fe = cv2.fisheye
    name = t
    if not name.startswith('CALIB_') and hasattr(fe, f'CALIB_{name}'):
        name = f'CALIB_{name}'
    if not hasattr(fe, name):
        names = sorted(n for n in dir(fe) if n.startswith('CALIB_'))
        hint = ', '.join(names[:12]) + ('...' if len(names) > 12 else '')
        raise argparse.ArgumentTypeError(
            f'Unknown cv2.fisheye flag {token!r}. Use a decimal, 0x..., or a name like '
            f'CALIB_FIX_K1. Examples: {hint}'
        )
    v = getattr(fe, name)
    if not isinstance(v, int):
        raise argparse.ArgumentTypeError(f'cv2.fisheye.{name} is not an integer; got {type(v).__name__}')
    return int(v)


from camera_charuco_calib.calib_sources.calib_source_common import use_gui_preview
from camera_charuco_calib.calib_sources.calib_source_file import run_file_source
from camera_charuco_calib.calib_sources.calib_source_ros2 import run_ros2_source
from camera_charuco_calib.calib_sources.calib_source_usb import run_usb_source
from camera_charuco_calib.charuco_calib_core import (
    ARUCO_DICTS,
    DEBUG_OVERLAY_SUBDIR,
    DEBUG_RAW_SUBDIR,
    CalibrationAccumulator,
    build_opencv_calib_flags,
    calibrate_from_accumulator,
    create_charuco_board,
    create_detector,
    get_aruco_dict,
    save_calibration_json,
    save_calibration_yaml,
    save_ros_usb_cam_camera_info_yaml,
)
from camera_charuco_calib.charuco_calib_fisheye import build_fisheye_calib_flags


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Calibrate camera using a ChArUco board.\n\n'
            'Input sources:\n'
            '  --source file : read images from disk glob(s)\n'
            '  --source usb  : capture frames from cv2.VideoCapture(camera_id);\n'
            '                  optional --capture_width/--capture_height/--capture_fps\n'
            '  --source ros2 : subscribe ROS2 Image topic via cv_bridge\n\n'
            'Board geometry defaults match charuco_board_gen.py:\n'
            '  -w / --width       5  (squares in X)\n'
            '  -H / --height      7  (squares in Y; -h is reserved for --help)\n'
            '  --aruco_dict       DICT_4X4_50\n\n'
            'PHYSICAL SIZES (no defaults — you must measure on the printed board):\n'
            '  --square_length    real side length of one chessboard square\n'
            '  --marker_size      real side length of one black ArUco marker\n'
            '  Use the same unit for both (e.g. mm or meters). Do not pass\n'
            '  charuco_board_gen.py --square_px / --marker_px here; those are\n'
            '  only pixel sizes for the PNG, not millimeters on paper.\n'
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        'images',
        nargs='*',
        help='image glob(s), only used when --source file, e.g. images/*.jpg'
    )

    parser.add_argument(
        '--source',
        type=str,
        default='file',
        choices=['file', 'usb', 'ros2'],
        help='input source type'
    )

    parser.add_argument(
        '-w', '--width',
        type=int,
        default=5,
        help='squares in X (default: 5; same as charuco_board_gen.py --width)'
    )
    parser.add_argument(
        '-H', '--height',
        type=int,
        default=7,
        help='squares in Y (default: 7; same as charuco_board_gen.py --height; use -H, not -h)'
    )
    parser.add_argument(
        '--aruco_dict',
        type=str,
        default='DICT_4X4_50',
        choices=sorted(ARUCO_DICTS.keys()),
        help='ArUco dictionary (default: DICT_4X4_50; same as charuco_board_gen.py)'
    )

    parser.add_argument(
        '--square_length',
        type=float,
        default=None,
        help=(
            'REQUIRED, measured on the physical print: side length of one chessboard square '
            '(same unit as --marker_size, e.g. mm). Not --square_px from charuco_board_gen.py.'
        )
    )
    parser.add_argument(
        '--marker_size',
        type=float,
        default=None,
        help=(
            'REQUIRED, measured on the physical print: side length of one black ArUco marker '
            '(same unit as --square_length). Not --marker_px from charuco_board_gen.py.'
        )
    )

    parser.add_argument(
        '--projection-model',
        dest='projection_model',
        type=str,
        default='pinhole',
        choices=['pinhole', 'fisheye'],
        help=(
            'pinhole: cv2.calibrateCamera. fisheye: cv2.fisheye.calibrate (requires 3D/2D points; '
            'use --distortion-model fisheye).'
        ),
    )
    parser.add_argument(
        '--distortion-model',
        dest='distortion_model',
        type=str,
        default='brown',
        choices=['brown', 'brown_rational', 'fisheye'],
        help=(
            'With pinhole: brown or brown_rational. With fisheye projection: must be fisheye '
            '(Kannala–Brandt / equidistant D, 4 coeffs).'
        ),
    )
    parser.add_argument(
        '--calib-extra-flags',
        dest='calib_extra_flags',
        type=parse_calib_extra_flags,
        default=0,
        help=(
            'Extra OpenCV calibrate flags ORed with --distortion-model: integer, 0x..., '
            'or cv2 constant name(s) e.g. CALIB_RATIONAL_MODEL or two: '
            'CALIB_RATIONAL_MODEL,CALIB_FIX_K3. Default: 0.'
        ),
    )
    parser.add_argument(
        '--calib-max-iter',
        dest='calib_max_iter',
        type=int,
        default=None,
        help='If set (optionally with --calib-epsilon), use this TermCriteria max_iter for calibration.',
    )
    parser.add_argument(
        '--calib-epsilon',
        dest='calib_epsilon',
        type=float,
        default=None,
        help='If set (optionally with --calib-max-iter), use this TermCriteria epsilon for calibration.',
    )
    parser.add_argument(
        '--fisheye-extra-flags',
        dest='fisheye_extra_flags',
        type=parse_fisheye_extra_flags,
        default=0,
        help=(
            'OR onto fisheye defaults (RECOMPUTE_EXTRINSIC|CHECK_COND|FIX_SKEW). '
            'Decimal, 0x..., or cv2.fisheye names e.g. CALIB_FIX_K1,CALIB_FIX_K2. '
            'Only for --projection-model fisheye.'
        ),
    )

    parser.add_argument(
        '--min_corners',
        type=int,
        default=6,
        help='minimum detected ChArUco corners per frame to keep it'
    )
    parser.add_argument(
        '--debug_images',
        action='store_true',
        help='write debug PNGs under --debug_dir: overlay/ and raw/ (default: off)',
    )
    parser.add_argument(
        '--debug_dir',
        type=str,
        default=os.path.join('~', 'amr_data', 'camera_calib_debug'),
        help='root directory when --debug_images (default: ~/amr_data/camera_calib_debug)',
    )
    parser.add_argument(
        '--output_yaml',
        type=str,
        default=os.path.join('~', 'amr_data', 'camera_config_for_opencv.yml'),
        help='output calibration YAML (OpenCV FileStorage)',
    )
    parser.add_argument(
        '--output_json',
        type=str,
        default=os.path.join('~', 'amr_data', 'camera_config.json'),
        help='output calibration JSON (legacy AMR format)'
    )
    parser.add_argument(
        '--no_output_json',
        action='store_true',
        help='do not write camera_config JSON'
    )
    parser.add_argument(
        '--output_camera_info_yaml',
        type=str,
        default=os.path.join('~', 'amr_data', 'camera_config_for_usb_cam_ros.yaml'),
        help='ROS usb_cam camera_info YAML (camera_calibration format)',
    )
    parser.add_argument(
        '--no_output_camera_info_yaml',
        action='store_true',
        help='do not write ROS usb_cam camera_info YAML',
    )
    parser.add_argument(
        '--camera_info_camera_name',
        type=str,
        default='usb_cam_color',
        help='camera_name in ROS camera_info YAML',
    )
    parser.add_argument(
        '--requested_width',
        type=int,
        default=None,
        help='optional stream width for JSON requested_width (default: calibrated image width)'
    )
    parser.add_argument(
        '--requested_height',
        type=int,
        default=None,
        help='optional stream height for JSON requested_height (default: calibrated image height)'
    )
    parser.add_argument(
        '--requested_fps',
        type=int,
        default=30,
        help='FPS field in JSON requested_fps (default: 30)'
    )
    parser.add_argument(
        '--save_captures_dir',
        type=str,
        default=os.path.join('~', 'amr_data', 'charuco_captures'),
        help='directory for accepted raw PNG frames in usb/ros2 mode',
    )
    parser.add_argument(
        '--no_save_captures',
        action='store_true',
        help='do not save accepted raw frames to disk (usb/ros2 only)'
    )

    # USB
    parser.add_argument(
        '--camera_id',
        type=int,
        default=0,
        help='camera index for usb mode (default: 0)'
    )
    parser.add_argument(
        '--usb_backend',
        type=str,
        default='v4l2',
        choices=['auto', 'v4l2', 'default'],
        help=(
            'VideoCapture API for --source usb. Default: v4l2 (matches bbox_gesture runtime). '
            'auto tries V4L2 first on Linux; default uses OpenCV default backend.'
        ),
    )
    parser.add_argument(
        '--use_mjpeg',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Set MJPEG FOURCC before width/height/FPS (default: ON, matching bbox_gesture runtime). '
            'Many UVC cameras only support high resolutions (e.g. 1280x720@30) in MJPEG mode. '
            'Use --no-use_mjpeg to force YUYV/default format.'
        ),
    )
    parser.add_argument(
        '--capture_width',
        type=int,
        default=None,
        metavar='W',
        help='--source usb: set cv2.CAP_PROP_FRAME_WIDTH after open',
    )
    parser.add_argument(
        '--capture_height',
        type=int,
        default=None,
        metavar='H',
        help='--source usb: set cv2.CAP_PROP_FRAME_HEIGHT after open',
    )
    parser.add_argument(
        '--capture_fps',
        type=float,
        default=None,
        metavar='FPS',
        help='--source usb: set cv2.CAP_PROP_FPS after open (optional)',
    )

    # ROS2
    parser.add_argument(
        '--topic',
        type=str,
        default='/camera/image_raw',
        help='ROS2 Image topic for ros2 mode'
    )

    parser.add_argument(
        '--capture_cooldown',
        type=float,
        default=0.5,
        help='minimum time gap between captures in live mode'
    )

    parser.add_argument(
        '--preview',
        type=str,
        default='auto',
        choices=['auto', 'gui', 'topic'],
        help=(
            'Live preview for usb/ros2: auto uses OpenCV window if DISPLAY is set, else publishes '
            'CompressedImage and uses Trigger services for capture/finish/abort'
        ),
    )
    parser.add_argument(
        '--preview_topic',
        type=str,
        default='/charuco_calib/preview/compressed',
        help='sensor_msgs/CompressedImage topic when --preview topic or auto without DISPLAY'
    )
    parser.add_argument(
        '--preview_jpeg_quality',
        type=int,
        default=85,
        help='JPEG quality 1-100 for preview topic (default: 85)'
    )
    parser.add_argument(
        '--ros_node_name',
        type=str,
        default='charuco_calib',
        help='ROS node name for headless services'
    )

    return parser


def validate_args(args):
    if args.square_length is None:
        raise ValueError(
            '--square_length is required (physical measurement on the printed board).\n'
            'Use a ruler/caliper on the real square — do not use charuco_board_gen.py --square_px.'
        )

    if args.marker_size is None:
        raise ValueError(
            '--marker_size is required (physical measurement on the printed board).\n'
            'Use a ruler/caliper on the real black marker — do not use charuco_board_gen.py --marker_px.'
        )

    if args.marker_size >= args.square_length:
        raise ValueError('--marker_size must be smaller than --square_length')

    if args.source == 'file' and len(args.images) == 0:
        raise ValueError('For --source file, provide image glob(s), e.g. "images/*.jpg"')

    if not (1 <= args.preview_jpeg_quality <= 100):
        raise ValueError('--preview_jpeg_quality must be between 1 and 100')

    if args.source == 'usb':
        if args.capture_width is not None and args.capture_width < 1:
            raise ValueError('--capture_width must be >= 1')
        if args.capture_height is not None and args.capture_height < 1:
            raise ValueError('--capture_height must be >= 1')
        if args.capture_fps is not None and args.capture_fps <= 0:
            raise ValueError('--capture_fps must be > 0')

    if int(args.calib_extra_flags) < 0:
        raise ValueError('--calib-extra-flags must be >= 0')
    if int(args.fisheye_extra_flags) < 0:
        raise ValueError('--fisheye-extra-flags must be >= 0')

    if args.projection_model == 'pinhole':
        if args.distortion_model not in ('brown', 'brown_rational'):
            raise ValueError(
                'With --projection-model pinhole, use --distortion-model brown or brown_rational (not fisheye).'
            )
    else:
        if args.distortion_model != 'fisheye':
            raise ValueError(
                'With --projection-model fisheye, set --distortion-model fisheye (OpenCV equidistant model).'
            )


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    if args.debug_images:
        args.debug_dir = os.path.expanduser(args.debug_dir)
    else:
        args.debug_dir = None

    args.output_yaml = os.path.expanduser(args.output_yaml)
    if not args.no_output_camera_info_yaml:
        args.output_camera_info_yaml = os.path.expanduser(args.output_camera_info_yaml)

    if args.source in ('usb', 'ros2'):
        if args.no_save_captures:
            args.save_captures_dir = None
        else:
            args.save_captures_dir = os.path.expanduser(args.save_captures_dir)
    else:
        args.save_captures_dir = None

    use_gui = True
    if args.source in ('usb', 'ros2'):
        use_gui = use_gui_preview(args)

    preview_note = ''
    if args.source in ('usb', 'ros2'):
        mode = 'OpenCV window' if use_gui else f"ROS {args.preview_topic} + Trigger services"
        preview_note = f' | preview={args.preview} ({mode})'
    if args.projection_model == 'pinhole':
        used_calib_flags = build_opencv_calib_flags(
            args.projection_model, args.distortion_model, int(args.calib_extra_flags)
        )
        used_fisheye_flags = None
        _flags_line = f'calib_flags={used_calib_flags} (0x{used_calib_flags:x})'
    else:
        used_calib_flags = None
        used_fisheye_flags = build_fisheye_calib_flags(int(args.fisheye_extra_flags))
        _flags_line = f'fisheye_flags={used_fisheye_flags} (0x{used_fisheye_flags:x})'
    print(
        f'[calibrate_camera_node] ChArUco calibrator. source={args.source}{preview_note} | '
        f'projection={args.projection_model} distortion={args.distortion_model} | {_flags_line}',
        flush=True,
    )

    if args.debug_dir is not None:
        os.makedirs(args.debug_dir, exist_ok=True)
        os.makedirs(os.path.join(args.debug_dir, DEBUG_OVERLAY_SUBDIR), exist_ok=True)
        os.makedirs(os.path.join(args.debug_dir, DEBUG_RAW_SUBDIR), exist_ok=True)
    if args.source in ('usb', 'ros2') and args.save_captures_dir:
        os.makedirs(args.save_captures_dir, exist_ok=True)

    aruco_dict = get_aruco_dict(args.aruco_dict)
    board, allow_charuco_detector = create_charuco_board(
        args.width,
        args.height,
        args.square_length,
        args.marker_size,
        aruco_dict,
    )
    detector = create_detector(board, allow_charuco_detector)

    acc = CalibrationAccumulator(
        board=board,
        min_corners=args.min_corners,
        debug_dir=args.debug_dir,
        save_captures_dir=args.save_captures_dir if args.source in ('usb', 'ros2') else None,
        squares_x=args.width,
        squares_y=args.height,
        square_len=args.square_length,
    )

    try:
        if args.source == 'file':
            run_file_source(args, board, aruco_dict, detector, acc)
        elif args.source == 'usb':
            run_usb_source(args, board, aruco_dict, detector, acc, use_gui)
        elif args.source == 'ros2':
            run_ros2_source(args, board, aruco_dict, detector, acc, use_gui)
        else:
            raise RuntimeError(f'Unknown source: {args.source}')

        rms, camera_matrix, dist_coeffs, _rvecs, _tvecs = calibrate_from_accumulator(
            acc,
            board,
            projection_model=args.projection_model,
            distortion_model=args.distortion_model,
            calib_flags_extra=int(args.calib_extra_flags),
            calib_max_iter=args.calib_max_iter,
            calib_epsilon=args.calib_epsilon,
            fisheye_flags_extra=int(args.fisheye_extra_flags),
        )

        print('\n===== Calibration Result =====')
        print('Source            :', args.source)
        print('Projection model  :', args.projection_model)
        print('Distortion model  :', args.distortion_model)
        if used_calib_flags is not None:
            print('OpenCV calib flags (calibrate3d):', used_calib_flags, f'(0x{used_calib_flags:x})')
        if used_fisheye_flags is not None:
            print('OpenCV fisheye flags          :', used_fisheye_flags, f'(0x{used_fisheye_flags:x})')
        print('Frames used     :', acc.kept)
        print('Image size      :', acc.image_size)
        print('Board squares X :', args.width)
        print('Board squares Y :', args.height)
        print('Square length   :', args.square_length)
        print('Marker size     :', args.marker_size)
        print('Aruco dict      :', args.aruco_dict)
        print('RMS             :', rms)
        print('Camera matrix:\n', camera_matrix)
        print('Dist coeffs:\n', dist_coeffs.ravel())

        save_calibration_yaml(
            args.output_yaml,
            acc.image_size,
            args,
            rms,
            camera_matrix,
            dist_coeffs,
            resolved_calib_flags=used_calib_flags,
            resolved_fisheye_flags=used_fisheye_flags,
        )
        print('\nSaved calibration to:', os.path.abspath(args.output_yaml))
        if not args.no_output_json:
            json_path = save_calibration_json(
                args.output_json,
                acc.image_size,
                camera_matrix,
                dist_coeffs,
                rms,
                requested_width=args.requested_width,
                requested_height=args.requested_height,
                requested_fps=args.requested_fps,
                projection_model=args.projection_model,
                distortion_model=args.distortion_model,
                calib_flags=used_calib_flags,
                fisheye_flags=used_fisheye_flags,
            )
            print('Saved camera config (JSON) to:', os.path.abspath(json_path))
        if not args.no_output_camera_info_yaml:
            _dm = 'equidistant' if args.projection_model == 'fisheye' else 'plumb_bob'
            ci_path = save_ros_usb_cam_camera_info_yaml(
                args.output_camera_info_yaml,
                acc.image_size,
                camera_matrix,
                dist_coeffs,
                camera_name=args.camera_info_camera_name,
                distortion_model=_dm,
                rms=float(rms),
            )
            print('Saved ROS usb_cam camera_info YAML to:', os.path.abspath(ci_path))
        if args.debug_dir:
            _dbg = os.path.abspath(args.debug_dir)
            print('Debug overlay (detection) saved in:', os.path.join(_dbg, DEBUG_OVERLAY_SUBDIR))
            print('Debug raw (offline calibration)  saved in:', os.path.join(_dbg, DEBUG_RAW_SUBDIR))
        if args.source in ('usb', 'ros2') and args.save_captures_dir:
            print('Accepted raw frames saved in:', os.path.abspath(args.save_captures_dir))

    except KeyboardInterrupt:
        print('\nAborted by user.')
    finally:
        if use_gui:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
