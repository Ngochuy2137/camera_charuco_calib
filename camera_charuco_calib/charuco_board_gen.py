#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a ChArUco board image for printing."""

import argparse
import os

import cv2

from camera_charuco_calib.charuco_calib_core import ARUCO_DICTS, get_aruco_dict


def _create_charuco_board(squares_x, squares_y, square_len, marker_len, aruco_dict):
    if hasattr(cv2.aruco, 'CharucoBoard_create'):
        return cv2.aruco.CharucoBoard_create(
            squares_x, squares_y, square_len, marker_len, aruco_dict
        )
    if hasattr(cv2.aruco, 'CharucoBoard'):
        return cv2.aruco.CharucoBoard(
            (squares_x, squares_y), square_len, marker_len, aruco_dict
        )
    raise RuntimeError('Your cv2.aruco does not support CharucoBoard.')


def _render_board_image(board, out_size, margin, border_bits):
    if hasattr(board, 'generateImage'):
        return board.generateImage(out_size, marginSize=margin, borderBits=border_bits)
    if hasattr(board, 'draw'):
        return board.draw(out_size, marginSize=margin, borderBits=border_bits)
    raise RuntimeError('This board object does not support generateImage/draw.')


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Generate a ChArUco board image for printing.\n\n'
            'Defaults for -w/-H/--aruco_dict match calibrate_camera_node.py board geometry.\n'
            '--square_px and --marker_px are only for the PNG resolution, not real-world units;\n'
            'after printing, measure the board and pass those sizes to calibrate_camera_node.py.'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('-w', '--width', type=int, default=5, help='squares in X (default: 5)')
    parser.add_argument('-H', '--height', type=int, default=7, help='squares in Y (default: 7)')
    parser.add_argument('--square_px', type=int, default=200, help='square size in output image pixels')
    parser.add_argument('--marker_px', type=int, default=120, help='marker size in output image pixels')
    parser.add_argument('--margin_px', type=int, default=20, help='margin in output image pixels')
    parser.add_argument(
        '--aruco_dict', type=str, default='DICT_4X4_50',
        choices=sorted(ARUCO_DICTS.keys()),
        help='ArUco dictionary (default: DICT_4X4_50)',
    )
    parser.add_argument('-o', '--output', type=str, default='charuco_board.png', help='output PNG path')
    args = parser.parse_args()

    if args.marker_px >= args.square_px:
        raise ValueError('--marker_px must be smaller than --square_px')

    aruco_dict = get_aruco_dict(args.aruco_dict)
    board = _create_charuco_board(
        args.width, args.height, float(args.square_px), float(args.marker_px), aruco_dict
    )

    img_w = args.width * args.square_px + 2 * args.margin_px
    img_h = args.height * args.square_px + 2 * args.margin_px

    img = _render_board_image(board, (img_w, img_h), args.margin_px, 1)

    ok = cv2.imwrite(args.output, img)
    if not ok:
        raise RuntimeError(f'Failed to save image: {args.output}')

    print('Saved board image to:', os.path.abspath(args.output))
    print('Print at 100% scale / Actual size.')
    print('After printing, measure the REAL square and marker sizes on paper.')
    print('Use those measured values in calibration, not these pixel values.')


if __name__ == '__main__':
    main()
