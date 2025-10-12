import cv2
import numpy as np
import cv2.aruco as aruco
import argparse
from pathlib import Path
import sys

# --- Configuration ---
WARP_WIDTH, WARP_HEIGHT = 800, 800
HSV_UNHEALTHY_LOWER = np.array([15, 60, 60])
HSV_UNHEALTHY_UPPER = np.array([35, 255, 255])
HSV_SOIL_BROWN_LOWER = np.array([10, 50, 20])
HSV_SOIL_BROWN_UPPER = np.array([30, 255, 200])


def normalize_hsv_to_target(image, target_hsv=(13, 66, 120), max_iter=3, tol=5):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    for _ in range(max_iter):
        mean_hsv = np.mean(hsv.reshape(-1, 3), axis=0)
        diff = np.array(target_hsv) - mean_hsv
        if np.all(np.abs(diff) < tol):
            break
        hsv += diff
        hsv = np.clip(hsv, [0, 0, 0], [179, 255, 255])
    mean_final = np.mean(hsv.reshape(-1, 3), axis=0)
    normalized_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return normalized_bgr, mean_hsv.astype(int), mean_final.astype(int)


def warp_aruco_board(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
    parameters = aruco.DetectorParameters_create()
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is None or len(ids) < 4:
        print("⚠ Not enough ArUco markers detected! Using original image as is.")
        return image, ids

    ids = ids.flatten()
    marker_dict = {id_: corner.reshape(4, 2) for id_, corner in zip(ids, corners)}
    required_ids = [80, 85, 90, 95]
    if not all(id_ in marker_dict for id_ in required_ids):
        print("⚠ Not all required ArUco IDs found! Using original image as is.")
        return image, ids

    top_left, top_right = marker_dict[80][0], marker_dict[85][1]
    bottom_right, bottom_left = marker_dict[90][2], marker_dict[95][3]
    src_pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    dst_pts = np.array([[0, 0], [WARP_WIDTH - 1, 0], [WARP_WIDTH - 1, WARP_HEIGHT - 1], [0, WARP_HEIGHT - 1]], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, matrix, (WARP_WIDTH, WARP_HEIGHT))
    return warped, ids


def plant_area(warped_image):
    mid_y = warped_image.shape[0] // 2
    top_half, bottom_half = warped_image[:mid_y, :], warped_image[mid_y:, :]
    mask_top = cv2.inRange(cv2.cvtColor(top_half, cv2.COLOR_BGR2HSV), HSV_SOIL_BROWN_LOWER, HSV_SOIL_BROWN_UPPER)
    mask_bottom = cv2.inRange(cv2.cvtColor(bottom_half, cv2.COLOR_BGR2HSV), HSV_SOIL_BROWN_LOWER, HSV_SOIL_BROWN_UPPER)
    return top_half if cv2.countNonZero(mask_top) > cv2.countNonZero(mask_bottom) else bottom_half


def plant_block(region):
    left = region[70:334, 68:324]
    right = region[70:334, 468:735]
    return left, right
