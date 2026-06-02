#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Live calibration frames from a ROS 2 sensor_msgs/Image topic."""

import sys
import threading
import time

import cv2
import numpy as np

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


class CharucoCalibRos2ImageNode:
    """Subscribe to sensor_msgs/Image; optional headless preview + services."""

    def __init__(self, image_topic: str, use_gui: bool, preview_topic: str, jpeg_quality: int, node_name: str):
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import CompressedImage, Image
        from cv_bridge import CvBridge

        rclpy.init(args=None)
        self._node = Node(node_name)
        self.use_gui = use_gui
        self._bridge = CvBridge()
        self.latest_frame = None
        self._jpeg_quality = jpeg_quality
        self._pub = None
        self._flags = HeadlessCmdFlags()
        self._node.create_subscription(Image, image_topic, self._image_cb, 10)
        if not use_gui:
            self._pub = self._node.create_publisher(CompressedImage, preview_topic, 1)
            _register_charuco_calib_services(self._node, self._flags)
            self._node.get_logger().info(
                f'Headless ROS2: input {image_topic} preview {preview_topic} | '
                f'services ~capture_frame ~finish_calibration ~abort_calibration (node {node_name})'
            )
        else:
            self._node.get_logger().info(f'ROS2 GUI: subscribing {image_topic}')
        self._shutdown_done = False
        self._spin_stop = None
        self._spin_thread = None
        if not use_gui:
            self._spin_stop = threading.Event()
            self._spin_thread = _start_rclpy_background_spin(
                self._node, self._spin_stop, 'charuco_ros2_spin'
            )

    @property
    def flags(self):
        return self._flags

    def _image_cb(self, msg):
        try:
            if msg.encoding in ('mono8', '8UC1'):
                gray = self._bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
                self.latest_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                self.latest_frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as err:
            self._node.get_logger().warn(f'cv_bridge conversion failed: {err}')

    @property
    def node(self):
        return self._node

    def publish_preview(self, vis_bgr):
        if self._pub is None:
            return
        _publish_preview_compressed(self._node, self._pub, vis_bgr, self._jpeg_quality)

    def poll_key(self):
        if self.use_gui:
            return cv2.waitKey(1) & 0xFF
        return _consume_headless_service_flags(self._flags)

    def spin_once(self):
        import rclpy

        rclpy.spin_once(self._node, timeout_sec=0.02)

    def shutdown(self):
        import rclpy

        if getattr(self, '_shutdown_done', False):
            return
        self._shutdown_done = True
        if self._spin_stop is not None:
            self._spin_stop.set()
        if self._spin_thread is not None:
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


def run_ros2_source(args, board, aruco_dict, detector, acc: CalibrationAccumulator, use_gui: bool):
    try:
        import rclpy
    except Exception as e:
        raise RuntimeError(
            'ROS2 mode requires rclpy, sensor_msgs, cv_bridge, std_srvs. '
            'Make sure ROS2 environment is sourced.'
        ) from e

    try:
        host = CharucoCalibRos2ImageNode(
            args.topic,
            use_gui,
            args.preview_topic,
            args.preview_jpeg_quality,
            args.ros_node_name,
        )
    except Exception as e:
        raise RuntimeError(
            'ROS2 mode failed to start (rclpy, sensor_msgs, cv_bridge, std_srvs). '
            'Source your ROS 2 workspace.'
        ) from e

    print('ROS2 mode:')
    print(f'  input topic: {args.topic}')
    if use_gui:
        print('  c or Enter (in window): capture  q: finish  ESC: quit')
    else:
        print(f'  preview: {args.preview_topic}')
        print(f'  services: /{args.ros_node_name}/capture_frame, finish_calibration, abort_calibration')
    if sys.stdin.isatty():
        print('  TTY: Enter = capture | q + Enter = finish | abort + Enter = quit')
    elif not use_gui:
        print(
            '  Note: stdin is not a TTY (typical under `ros2 launch`). Enter here does nothing — '
            'use the Trigger services above, or run the calibration node directly in this shell.',
            flush=True,
        )

    frame_idx = 0
    last_capture_time = 0.0

    try:
        while rclpy.ok():
            if use_gui:
                host.spin_once()

            if host.latest_frame is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, f'Waiting for topic: {args.topic}', (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                if use_gui:
                    cv2.imshow('charuco_live', blank)
                    key = _merge_live_keys(cv2.waitKey(1) & 0xFF)
                else:
                    host.publish_preview(blank)
                    key = _merge_live_keys(_consume_headless_service_flags(host.flags))
                if key == 27:
                    print('[calib] abort (ESC, abort service, or stdin)', flush=True)
                    raise KeyboardInterrupt
                if key == ord('q'):
                    print('[calib] finish_calibration: running solve...', flush=True)
                    break
                continue

            frame_bgr = host.latest_frame.copy()
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
                f'source=ros2 topic={args.topic}',
                f'corners={num_corners} kept={acc.kept}',
                'c/Enter=capture  q=finish  esc=quit' if use_gui else 'TTY:Enter=capture | svc capture_frame',
            ]
            vis = draw_live_overlay(frame_bgr, marker_corners, marker_ids, charuco_corners, charuco_ids, info_text)
            if use_gui:
                cv2.imshow('charuco_live', vis)
                key = _merge_live_keys(cv2.waitKey(1) & 0xFF)
            else:
                host.publish_preview(vis)
                key = _merge_live_keys(_consume_headless_service_flags(host.flags))

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
                frame_name = f'ros2_{frame_idx:04d}'
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
        host.shutdown()
        if use_gui:
            cv2.destroyAllWindows()
