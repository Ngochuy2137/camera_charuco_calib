import os
from glob import glob

from setuptools import setup, find_packages

package_name = 'camera_charuco_calib'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='ChArUco camera calibration tools',
    license='TODO',
    entry_points={
        'console_scripts': [
            'calibrate_camera_node.py = camera_charuco_calib.calibrate_camera_node:main',
            'charuco_board_gen.py = camera_charuco_calib.charuco_board_gen:main',
        ],
    },
)
