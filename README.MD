# camera_charuco_calib

ChArUco calibration for **fisheye UVC cameras** (board generator + live/offline calibrator).

Uses `cv2.fisheye.calibrate` (Kannala–Brandt / equidistant, 4 distortion coeffs).  
ROS output: `distortion_model: equidistant` in `camera_config_for_usb_cam_ros.yaml`.

```bash
ros2 run camera_charuco_calib charuco_board_gen.py -o charuco_board.png
ros2 run camera_charuco_calib calibrate_camera_node.py ...
```

## 1. Print the board

Default geometry: **5×7** squares, dictionary **`DICT_4X4_50`**.

1. Generate PNG, print at **100% scale** (no fit-to-page).
2. Measure on paper: **square side** → `--square_length`, **marker side** → `--marker_size` (same unit, e.g. cm).
3. Do **not** use `--square_px` / `--marker_px` from the generator — those are PNG pixels only.

## 2. Calibrate fisheye (USB / follow-me camera)

Use the **same resolution and pixel format** as runtime (e.g. `bbox_gesture` at 1280×720 MJPEG).

USB defaults: **V4L2**, **MJPEG on** (matches `bbox_gesture`). Verify in log:

```text
[usb] capture props: ... -> reported 1280x720 @ 30.00 fps (fourcc='MJPG')
```

If you see a lower resolution (e.g. 800×600) or `fourcc='YUYV'`, calibration FOV will not match runtime.

**Interactive** (`ros2 run`, Enter = capture):

```bash
ros2 run camera_charuco_calib calibrate_camera_node.py --source usb \
    --capture_width 1280 --capture_height 720 --capture_fps 30 \
    --projection-model fisheye --distortion-model fisheye \
    --square_length 3.88 --marker_size 2.8 --preview topic \
    --debug_images \
    --camera_id 0
```

Fisheye needs **≥5 kept frames** with valid ChArUco 3D/2D point pairs. Collect **15–30** varied viewpoints (center, edges, tilt).

Other sources:

| `--source` | Use when |
|------------|----------|
| `file` | Offline from saved images: `'frames/*.png'` |
| `usb` | Direct OpenCV capture (Jetson / USB fisheye webcam) |
| `ros2` | Subscribe `sensor_msgs/Image` on `--topic` |

```bash
# Offline (add fisheye flags)
ros2 run camera_charuco_calib calibrate_camera_node.py --source file 'calib_frames/*.png' \
    --projection-model fisheye --distortion-model fisheye \
    --square_length 3.88 --marker_size 2.8

# From another node's Image topic
ros2 run camera_charuco_calib calibrate_camera_node.py --source usb \
    --capture_width 1280 --capture_height 720 --capture_fps 30 \
    --projection-model fisheye --distortion-model fisheye \
    --square_length 3.88 --marker_size 2.8 --preview topic \
    --debug_images \
    --camera_id 0
```

Pinhole cameras (non-fisheye): use `--projection-model pinhole --distortion-model brown_rational`.

## 3. Capture controls (USB / ROS2 live)

**`ros2 run` (interactive):** stdin is a TTY → **Enter** = capture, **q** + Enter = finish, **abort** + Enter = quit.

**`ros2 launch` (headless):** stdin is NOT a TTY → Enter does nothing. Use Trigger services from another terminal:

```bash
ros2 service call /charuco_calib/capture_frame std_srvs/srv/Trigger {}
ros2 service call /charuco_calib/finish_calibration std_srvs/srv/Trigger {}
ros2 service call /charuco_calib/abort_calibration std_srvs/srv/Trigger {}
```

| `--preview` | Behavior |
|-------------|----------|
| `auto` | GUI if `DISPLAY` set, else headless |
| `gui` | OpenCV window: `c`/Enter capture, `q` finish, ESC abort |
| `topic` | Preview on `/charuco_calib/preview/compressed` + Trigger services |

## 4. Outputs

| File | Default path | Notes |
|------|----------------|-------|
| OpenCV YAML | `~/amr_data/camera_config_for_opencv.yml` | K, D (4 coeffs), RMS |
| ROS CameraInfo YAML | `~/amr_data/camera_config_for_usb_cam_ros.yaml` | `distortion_model: equidistant` |
| AMR JSON | `~/amr_data/camera_config.json` | Legacy format |

Use the ROS YAML with `camera_static.launch.py` / `camera_info_from_yaml_node`.

Optional: `--debug_images` → `~/amr_data/camera_calib_debug/overlay/` and `raw/`.

## 5. Launch (headless, fisheye defaults)

```bash
# Terminal 1
ros2 launch camera_charuco_calib calibrate_camera.launch.py \
    square_length:=3.88 marker_size:=2.8 camera_id:=0

# Terminal 2: capture frames, then finish
ros2 service call /charuco_calib/capture_frame std_srvs/srv/Trigger {}
ros2 service call /charuco_calib/finish_calibration std_srvs/srv/Trigger {}
```

Override model if needed: `projection_model:=pinhole distortion_model:=brown_rational`.

For interactive capture with Enter key, use `ros2 run` directly (see §2).
