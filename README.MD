# camera_charuco_calib

ChArUco camera calibration (board generator + live/offline calibrator).

```bash
ros2 run camera_charuco_calib charuco_board_gen.py -o charuco_board.png
ros2 run camera_charuco_calib calibrate_camera_node.py ...
```

## 1. Print the board

Default geometry: **5×7** squares, dictionary **`DICT_4X4_50`**.

1. Generate PNG, print at **100% scale** (no fit-to-page).
2. Measure on paper: **square side** → `--square_length`, **marker side** → `--marker_size` (same unit, e.g. cm).
3. Do **not** use `--square_px` / `--marker_px` from the generator — those are PNG pixels only.

## 2. Calibrate (USB / follow-me camera)

Use the **same resolution and pixel format** as runtime (e.g. `bbox_gesture` at 1280×720 MJPEG).

USB mode defaults: **V4L2** backend, **MJPEG on** (matches `bbox_gesture`). Check the log line:

```text
[usb] capture props: ... -> reported 1280x720 @ 30.00 fps (fourcc='MJPG')
```

If you see a lower resolution (e.g. 800×600) or `fourcc='YUYV'`, calibration FOV will not match runtime.

```bash
ros2 run camera_charuco_calib calibrate_camera_node.py --source usb \
    --capture_width 1280 --capture_height 720 --capture_fps 30 \
    --projection-model pinhole --distortion-model brown_rational \
    --square_length 3.88 --marker_size 2.8 --preview topic \
    --debug_images \
    --camera_id 0
```

Pinhole + rational Brown (typical UVC): `--distortion-model brown_rational`.  
Fisheye: `--projection-model fisheye --distortion-model fisheye`.

Other sources:

| `--source` | Use when |
|------------|----------|
| `file` | Offline from saved images: `'frames/*.png'` |
| `usb` | Direct OpenCV capture (Jetson / USB webcam) |
| `ros2` | Subscribe `sensor_msgs/Image` on `--topic` |

```bash
# Offline
ros2 run camera_charuco_calib calibrate_camera_node.py --source file 'calib_frames/*.png' \
    --square_length 3.88 --marker_size 2.8

# From another node's Image topic
ros2 run camera_charuco_calib calibrate_camera_node.py --source ros2 --topic /camera/image_raw \
    --square_length 3.88 --marker_size 2.8 --preview topic
```

**Pi CSI via `camera` package:** `camera_node` must publish raw `Image` (`camera_stream_compressed:=false`) on `/camera/camera_stream/image`. See `camera/README.MD` for enabling the stream.

## 3. Capture controls (USB / ROS2 live)

| `--preview` | Behavior |
|-------------|----------|
| `auto` | GUI if `DISPLAY` set, else headless |
| `gui` | OpenCV window: `c`/Enter capture, `q` finish, ESC abort |
| `topic` | Preview on `/charuco_calib/preview/compressed` + Trigger services |

Headless services (default node name `charuco_calib`):

```bash
ros2 service call /charuco_calib/capture_frame std_srvs/srv/Trigger {}
ros2 service call /charuco_calib/finish_calibration std_srvs/srv/Trigger {}
ros2 service call /charuco_calib/abort_calibration std_srvs/srv/Trigger {}
```

Collect **15–30** varied viewpoints before `finish_calibration`.

## 4. Outputs

| File | Default path |
|------|----------------|
| OpenCV YAML | `~/amr_data/camera_config_for_opencv.yml` |
| ROS CameraInfo YAML | `~/amr_data/camera_config_for_usb_cam_ros.yaml` |
| AMR JSON | `~/amr_data/camera_config.json` |

Use the ROS YAML with `camera_static.launch.py` / `camera_info_from_yaml_node` (`camera_info_url:=file://$HOME/amr_data/camera_config_for_usb_cam_ros.yaml`).

Optional: `--debug_images` → `~/amr_data/camera_calib_debug/overlay/` and `raw/`.

## 5. Launch

```bash
ros2 launch camera_charuco_calib calibrate_camera.launch.py \
    square_length:=3.88 marker_size:=2.8 camera_id:=0
```

Override capture size via extra args if needed (edit launch or run the node directly).
