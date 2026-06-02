#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Live calibration frames from a USB camera via cv2.VideoCapture.

Camera open sequence mirrors bbox_gesture's OpencvCameraReader to avoid
FOURCC/resolution mismatches between calibration and runtime.
"""

import sys
import threading
import time

import cv2

from .calib_source_common import (
    HeadlessCmdFlags,
    _consume_headless_service_flags,
    _merge_live_keys,
    _publish_preview_compressed,
    _register_charuco_calib_services,
    _start_rclpy_background_spin,
)
from camera_charuco_calib.charuco_calib_core import (
    CalibrationAccumulator,
    detect_charuco_compatible,
    draw_live_overlay,
    ensure_gray,
)


class CharucoHeadlessUsbNode:
    """rclpy node: preview publisher + Trigger services (USB, no display)."""

    def __init__(self, node_name: str, preview_topic: str, jpeg_quality: int):
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import CompressedImage

        rclpy.init(args=None)
        self._node = Node(node_name)
        self._flags = HeadlessCmdFlags()
        self._pub = self._node.create_publisher(CompressedImage, preview_topic, 1)
        self._jpeg_quality = jpeg_quality
        _register_charuco_calib_services(self._node, self._flags)
        self._node.get_logger().info(
            f'Headless USB: preview topic {preview_topic} | '
            f'services /{node_name}/capture_frame, /{node_name}/finish_calibration, /{node_name}/abort_calibration'
        )
        self._shutdown_done = False
        self._spin_stop = threading.Event()
        self._spin_thread = _start_rclpy_background_spin(self._node, self._spin_stop, 'charuco_usb_spin')

    @property
    def node(self):
        return self._node

    @property
    def flags(self):
        return self._flags

    def publish_preview(self, vis_bgr):
        _publish_preview_compressed(self._node, self._pub, vis_bgr, self._jpeg_quality)

    def poll_key(self):
        return _consume_headless_service_flags(self._flags)

    def shutdown(self):
        import rclpy

        if self._shutdown_done:
            return
        self._shutdown_done = True
        self._spin_stop.set()
        try:
            self._spin_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            self._node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


def open_usb_camera(camera_id: int, backend: str):
    """
    Open a USB camera. On many Linux ARM boards (e.g. Jetson), the default OpenCV
    backend picks GStreamer and can warn or segfault; V4L2 is usually stable.
    backend: 'auto' | 'v4l2' | 'default'
    """
    v4l2 = getattr(cv2, 'CAP_V4L2', None)

    if backend == 'v4l2':
        if v4l2 is None:
            raise RuntimeError('CAP_V4L2 is not available in this OpenCV build')
        cap = cv2.VideoCapture(camera_id, v4l2)
        return cap

    if backend == 'default':
        return cv2.VideoCapture(camera_id)

    # auto: prefer V4L2 on Linux (matches bbox_gesture OpencvCameraReader)
    if sys.platform.startswith('linux') and v4l2 is not None:
        cap = cv2.VideoCapture(camera_id, v4l2)
        if cap.isOpened():
            return cap
        cap.release()
        print('[WARN] USB camera: V4L2 failed to open; retrying default backend', flush=True)
    return cv2.VideoCapture(camera_id)


def configure_usb_capture(cap, args) -> None:
    """Apply FOURCC/resolution/FPS via OpenCV.

    Order matches bbox_gesture's OpencvCameraReader: FOURCC first, then W/H/FPS.
    On V4L2 UVC, setting FOURCC after resolution can cause the driver to negotiate
    a different pixel format that lacks the requested resolution.
    """
    use_mjpeg = getattr(args, 'use_mjpeg', True)
    if use_mjpeg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        print('[usb] FOURCC set to MJPEG (matching bbox_gesture runtime)', flush=True)
    if args.capture_width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.capture_width))
    if args.capture_height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.capture_height))
    if args.capture_fps is not None:
        cap.set(cv2.CAP_PROP_FPS, float(args.capture_fps))
    if args.capture_width is not None or args.capture_height is not None or args.capture_fps is not None:
        aw = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        ah = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        af = cap.get(cv2.CAP_PROP_FPS)
        af_str = f'{af:.2f}' if af == af and af > 0 else str(af)
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = ''.join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4))
        print(
            f'[usb] capture props: requested width={args.capture_width} height={args.capture_height} '
            f'fps={args.capture_fps} -> reported {aw}x{ah} @ {af_str} fps (fourcc={fourcc_str!r})',
            flush=True,
        )
        req_w = args.capture_width
        req_h = args.capture_height
        if req_w is not None and req_h is not None and (aw != req_w or ah != req_h):
            print(
                f'\033[33m[WARN] Driver negotiated {aw}x{ah} instead of requested {req_w}x{req_h}!\033[0m\n'
                f'       The camera may not support {req_w}x{req_h} in the current pixel format ({fourcc_str}).\n'
                f'       Run: v4l2-ctl --device=/dev/video<N> --list-formats-ext',
                flush=True,
            )


def run_usb_source(args, board, aruco_dict, detector, acc: CalibrationAccumulator, use_gui: bool):
    cap = open_usb_camera(args.camera_id, args.usb_backend)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open USB camera index {args.camera_id}')
    configure_usb_capture(cap, args)

    headless = None
    if not use_gui:
        try:
            headless = CharucoHeadlessUsbNode(
                args.ros_node_name,
                args.preview_topic,
                args.preview_jpeg_quality,
            )
        except Exception as e:
            cap.release()
            raise RuntimeError(
                'Headless USB mode requires rclpy, sensor_msgs, std_srvs. '
                'Source your ROS 2 workspace, or use --preview gui with DISPLAY set.'
            ) from e

    print('USB mode:')
    if use_gui:
        print('  c or Enter (in window): capture current frame if detection is good')
        print('  q: finish and calibrate')
        print('  ESC: quit without calibrating')
    else:
        print(f'  preview: {args.preview_topic} (sensor_msgs/CompressedImage, jpeg)')
        print(f'  ros2 service call /{args.ros_node_name}/capture_frame std_srvs/srv/Trigger "{{}}"')
        print(f'  ros2 service call /{args.ros_node_name}/finish_calibration std_srvs/srv/Trigger "{{}}"')
        print(f'  ros2 service call /{args.ros_node_name}/abort_calibration std_srvs/srv/Trigger "{{}}"')
    if sys.stdin.isatty():
        print('  TTY (this terminal): Enter = capture | q + Enter = finish | abort + Enter = quit')
    elif not use_gui:
        print(
            '  Note: stdin is not a TTY (typical under `ros2 launch`). Enter here does nothing — '
            'use the Trigger services above, or run the calibration node directly in this shell.',
            flush=True,
        )

    frame_idx = 0
    last_capture_time = 0.0

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                print('[WARN] Failed to read frame from USB camera')
                if headless is not None:
                    k = _merge_live_keys(headless.poll_key())
                else:
                    k = _merge_live_keys(0)
                if k == 27:
                    raise KeyboardInterrupt
                if k == ord('q'):
                    break
                continue

            gray = ensure_gray(frame_bgr)

            try:
                charuco_corners, charuco_ids, marker_corners, marker_ids = detect_charuco_compatible(
                    gray, board, aruco_dict, detector
                )
            except Exception as e:
                print(f'[WARN] Detection error: {e}')
                charuco_corners, charuco_ids, marker_corners, marker_ids = None, None, [], []

            num_corners = 0 if charuco_ids is None else len(charuco_ids)
            info_text = [
                f'source=usb camera_id={args.camera_id}',
                f'corners={num_corners} kept={acc.kept}',
                'c/Enter=capture  q=finish  esc=quit' if use_gui else 'TTY:Enter=capture | svc capture_frame',
            ]
            vis = draw_live_overlay(frame_bgr, marker_corners, marker_ids, charuco_corners, charuco_ids, info_text)

            if use_gui:
                cv2.imshow('charuco_live', vis)
                key = _merge_live_keys(cv2.waitKey(1) & 0xFF)
            else:
                headless.publish_preview(vis)
                key = _merge_live_keys(headless.poll_key())

            if key == 27:
                print('[calib] abort (ESC, abort service, or stdin)', flush=True)
                raise KeyboardInterrupt
            elif key == ord('q'):
                print('[calib] finish_calibration: running solve...', flush=True)
                break
            elif key == ord('c'):
                now = time.time()
                if now - last_capture_time < args.capture_cooldown:
                    print('[calib] capture ignored (cooldown)', flush=True)
                    continue
                frame_name = f'usb_{frame_idx:04d}'
                ok = acc.try_add_frame(frame_bgr, gray, frame_name, marker_corners, marker_ids, charuco_corners, charuco_ids)
                if ok:
                    print(f'[calib] captured {frame_name} (kept={acc.kept})', flush=True)
                    frame_idx += 1
                    last_capture_time = now
                else:
                    _rej = '[calib] capture rejected (corners/size)'
                    if args.debug_dir:
                        _rej += '; see debug_dir'
                    print(_rej, flush=True)
    finally:
        cap.release()
        if headless is not None:
            headless.shutdown()
        if use_gui:
            cv2.destroyAllWindows()
