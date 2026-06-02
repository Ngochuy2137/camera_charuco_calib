#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for live calibration sources (GUI vs headless preview and input mapping)."""

import os
import sys
import threading
import time

import cv2


def use_gui_preview(args) -> bool:
    """GUI (OpenCV window) vs headless (ROS CompressedImage + Trigger services)."""
    if args.preview == 'gui':
        return True
    if args.preview == 'topic':
        return False
    return bool(os.environ.get('DISPLAY', '').strip())


class HeadlessCmdFlags:
    __slots__ = ('capture', 'finish', 'abort')

    def __init__(self):
        self.capture = False
        self.finish = False
        self.abort = False


def _register_charuco_calib_services(node, flags: HeadlessCmdFlags):
    from std_srvs.srv import Trigger

    def _capture_cb(_req, resp):
        flags.capture = True
        node.get_logger().info('Service capture_frame received (queued for main loop)')
        print('[calib] service capture_frame -> queued', flush=True)
        resp.success = True
        resp.message = 'Capture queued'
        return resp

    def _finish_cb(_req, resp):
        flags.finish = True
        node.get_logger().info('Service finish_calibration received (queued)')
        print('[calib] service finish_calibration -> queued', flush=True)
        resp.success = True
        resp.message = 'Finish queued'
        return resp

    def _abort_cb(_req, resp):
        flags.abort = True
        node.get_logger().info('Service abort_calibration received (queued)')
        print('[calib] service abort_calibration -> queued', flush=True)
        resp.success = True
        resp.message = 'Abort queued'
        return resp

    node.create_service(Trigger, '~/capture_frame', _capture_cb)
    node.create_service(Trigger, '~/finish_calibration', _finish_cb)
    node.create_service(Trigger, '~/abort_calibration', _abort_cb)


def _publish_preview_compressed(node, publisher, vis_bgr, jpeg_quality: int):
    from sensor_msgs.msg import CompressedImage

    msg = CompressedImage()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.format = 'jpeg'
    ok, buf = cv2.imencode('.jpg', vis_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        return
    msg.data = buf.tobytes()
    publisher.publish(msg)


def _drain_rclpy_node(node, max_iter: int = 24):
    """Process pending subscriptions and service calls so clients do not hang."""
    import rclpy
    from rclpy.executors import ExternalShutdownException

    for _ in range(max_iter):
        try:
            rclpy.spin_once(node, timeout_sec=0.001)
        except ExternalShutdownException:
            break


def _start_rclpy_background_spin(node, stop_event: threading.Event, name: str = 'charuco_calib_spin'):
    def _loop():
        import rclpy

        while not stop_event.is_set():
            try:
                if not rclpy.ok():
                    break
            except Exception:
                break
            _drain_rclpy_node(node, 12)
            time.sleep(0.004)

    th = threading.Thread(target=_loop, daemon=True, name=name)
    th.start()
    return th


def _consume_headless_service_flags(flags: HeadlessCmdFlags) -> int:
    if flags.abort:
        flags.abort = False
        return 27
    if flags.finish:
        flags.finish = False
        return ord('q')
    if flags.capture:
        flags.capture = False
        return ord('c')
    return 0


def _tty_line_command_if_any() -> int:
    if not sys.stdin.isatty():
        return 0
    if sys.platform == 'win32':
        return 0
    try:
        import select
    except ImportError:
        return 0
    try:
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        if not readable:
            return 0
        line = sys.stdin.readline()
    except (ValueError, OSError):
        return 0
    s = line.strip().lower()
    if s == '':
        return ord('c')
    if s in ('q', 'quit', 'finish'):
        return ord('q')
    if s in ('abort', 'a'):
        return 27
    if s in ('c', 'capture', 'cap'):
        return ord('c')
    return 0


def _merge_live_keys(key: int) -> int:
    """Map GUI Enter to capture; merge TTY line commands (Enter / q / abort)."""
    if key in (13, 10):
        return ord('c')
    tty_key = _tty_line_command_if_any()
    if tty_key != 0:
        label = {ord('c'): 'capture', ord('q'): 'finish', 27: 'abort'}.get(tty_key, str(tty_key))
        print(f'[calib] stdin -> {label}', flush=True)
        return tty_key
    return key
