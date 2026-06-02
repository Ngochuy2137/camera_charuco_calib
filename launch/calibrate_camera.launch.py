"""Launch ChArUco fisheye camera calibration with common defaults."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    home = EnvironmentVariable('HOME')

    output_yaml = PathJoinSubstitution([home, 'amr_data', 'camera_config_for_opencv.yml'])
    output_json = PathJoinSubstitution([home, 'amr_data', 'camera_config.json'])
    save_captures = PathJoinSubstitution([home, 'amr_data', 'charuco_captures'])
    camera_info_yaml = PathJoinSubstitution([home, 'amr_data', 'camera_config_for_usb_cam_ros.yaml'])

    return LaunchDescription([
        DeclareLaunchArgument('source', default_value='usb', description='usb | ros2 | file'),
        DeclareLaunchArgument('square_length', default_value='3.88'),
        DeclareLaunchArgument('marker_size', default_value='2.8'),
        DeclareLaunchArgument('usb_backend', default_value='v4l2'),
        DeclareLaunchArgument('camera_id', default_value='0'),
        DeclareLaunchArgument('capture_width', default_value='1280'),
        DeclareLaunchArgument('capture_height', default_value='720'),
        DeclareLaunchArgument('capture_fps', default_value='30'),
        DeclareLaunchArgument('projection_model', default_value='fisheye', description='fisheye | pinhole'),
        DeclareLaunchArgument('distortion_model', default_value='fisheye', description='fisheye (with fisheye projection) | brown | brown_rational'),
        DeclareLaunchArgument('preview', default_value='topic', description='auto | gui | topic'),
        DeclareLaunchArgument('preview_topic', default_value='/charuco_calib/preview/compressed'),
        DeclareLaunchArgument('preview_jpeg_quality', default_value='85'),
        DeclareLaunchArgument('ros_node_name', default_value='charuco_calib'),
        DeclareLaunchArgument('image_topic', default_value='/camera/image_raw'),
        DeclareLaunchArgument('width', default_value='5', description='ChArUco squares X'),
        DeclareLaunchArgument('height', default_value='7', description='ChArUco squares Y'),
        DeclareLaunchArgument('aruco_dict', default_value='DICT_4X4_50'),
        DeclareLaunchArgument('output_yaml', default_value=output_yaml),
        DeclareLaunchArgument('output_json', default_value=output_json),
        DeclareLaunchArgument('save_captures_dir', default_value=save_captures),
        DeclareLaunchArgument('output_camera_info_yaml', default_value=camera_info_yaml),
        DeclareLaunchArgument('camera_info_camera_name', default_value='usb_cam_color'),
        ExecuteProcess(
            cmd=[
                'ros2', 'run', 'camera_charuco_calib', 'calibrate_camera_node.py',
                '--source', LaunchConfiguration('source'),
                '--square_length', LaunchConfiguration('square_length'),
                '--marker_size', LaunchConfiguration('marker_size'),
                '--usb_backend', LaunchConfiguration('usb_backend'),
                '--camera_id', LaunchConfiguration('camera_id'),
                '--capture_width', LaunchConfiguration('capture_width'),
                '--capture_height', LaunchConfiguration('capture_height'),
                '--capture_fps', LaunchConfiguration('capture_fps'),
                '--projection-model', LaunchConfiguration('projection_model'),
                '--distortion-model', LaunchConfiguration('distortion_model'),
                '--preview', LaunchConfiguration('preview'),
                '--preview_topic', LaunchConfiguration('preview_topic'),
                '--preview_jpeg_quality', LaunchConfiguration('preview_jpeg_quality'),
                '--ros_node_name', LaunchConfiguration('ros_node_name'),
                '--topic', LaunchConfiguration('image_topic'),
                '-w', LaunchConfiguration('width'),
                '-H', LaunchConfiguration('height'),
                '--aruco_dict', LaunchConfiguration('aruco_dict'),
                '--debug_images',
                '--output_yaml', LaunchConfiguration('output_yaml'),
                '--output_json', LaunchConfiguration('output_json'),
                '--save_captures_dir', LaunchConfiguration('save_captures_dir'),
                '--output_camera_info_yaml', LaunchConfiguration('output_camera_info_yaml'),
                '--camera_info_camera_name', LaunchConfiguration('camera_info_camera_name'),
            ],
            output='screen',
            shell=False,
        ),
    ])
