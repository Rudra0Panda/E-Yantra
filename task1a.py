'''
# Team ID:          eYRC#1894
# Theme:            KrishiDrone
# Author List:      1.Rudra Narayan Panda
#                   2.Mahasangram Kar
#                   3.Sidhartha Kumar Nayak
#                   4.Subham Kumar Rana
# Filename:         task1a.py
# Functions:        find_plant_area, split_area_into_plants, get_infection_score, Detection, main
# Global variables: ID_TO_POS, WIDTH, HEIGHT, BLOCK_ROWS, BLOCK_COLS, HSV_YELLOW_RANGE, HSV_BROWN_RANGE
'''

import cv2
import numpy as np
import cv2.aruco as aruco
import argparse
from pathlib import Path
import sys

# -- Configurations --
ID_TO_POS = {85: "tl", 90: "tr", 80: "bl", 95: "br"}
WIDTH, HEIGHT = 640, 500
BLOCK_ROWS, BLOCK_COLS = 3, 2
HSV_YELLOW_RANGE = (np.array([20, 100, 100]), np.array([35, 255, 255]))
HSV_BROWN_RANGE = (np.array([10, 60, 40]), np.array([30, 255, 200]))


def find_plant_area(roi):
    '''
    Purpose:
    ---
    Finds the PLANT area by determining which half of the ROI
    contains the Most amount of brown "soil" pixels.
    '''
    # 1. Divide the ROI into left and right halves.
    roi_h, roi_w = roi.shape[:2]
    midpoint = roi_w // 2
    left_half = roi[:, :midpoint]
    right_half = roi[:, midpoint:]

    # 2. Analyze the right half for soil content.
    hsv_right = cv2.cvtColor(right_half, cv2.COLOR_BGR2HSV)
    mask_right = cv2.inRange(hsv_right, HSV_BROWN_RANGE[0], HSV_BROWN_RANGE[1])
    right_score = cv2.countNonZero(mask_right)

    # 3. Analyze the left half for soil content.
    hsv_left = cv2.cvtColor(left_half, cv2.COLOR_BGR2HSV)
    mask_left = cv2.inRange(hsv_left, HSV_BROWN_RANGE[0], HSV_BROWN_RANGE[1])
    left_score = cv2.countNonZero(mask_left)

    # 4. Compare scores
    if right_score > left_score:
        return left_half
    else:
        return right_half


def split_area_into_plants(area, rows=3, cols=2):
    '''
    Purpose:
    ---
    Splits a given image area into a grid of smaller blocks and assigns a
    pre-defined label to each block.
    '''
    h, w = area.shape[:2]
    block_h, block_w = h // rows, w // cols
    labels = [["A", "D"], ["B", "E"], ["C", "F"]]
    blocks = []

    for c in range(cols):
        for r in range(rows):
            block = area[r * block_h:(r + 1) * block_h, c * block_w:(c + 1) * block_w]
            label = labels[r][c]
            blocks.append((label, block))
    return blocks


def get_infection_score(block, hsv_range):
    '''
    Purpose:
    ---
    Calculates the number of "infected" pixels in an image block.
    '''
    hsv = cv2.cvtColor(block, cv2.COLOR_BGR2HSV)
    lower, upper = hsv_range
    mask = cv2.inRange(hsv, lower, upper)
    return cv2.countNonZero(mask)


def Detection(image_path):
    '''
    Purpose:
    ---
    This function executes the entire pipeline from image loading to analysis
    and writes the final results to a text file.
    '''
    # 1. Load and Prepare the Image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    image = cv2.resize(image, (640, 480))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Enhance Image for Better Detection
    processed_gray = gray
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    processed_gray = clahe.apply(processed_gray)

    # 3. Detect ArUco Markers
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
    parameters = aruco.DetectorParameters()
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(processed_gray)

    if ids is None:
        raise RuntimeError("No ArUco markers detected!")

    ids = ids.flatten()

    # 4. Map Source and Destination Points
    pts_src = np.zeros((4, 2), dtype="float32")
    for corner, marker_id in zip(corners, ids):
        c = corner[0]
        cx, cy = np.mean(c, axis=0)
        if marker_id in ID_TO_POS:
            pos = ID_TO_POS[marker_id]
            if pos == "tl": pts_src[0] = [cx, cy]
            elif pos == "tr": pts_src[1] = [cx, cy]
            elif pos == "bl": pts_src[2] = [cx, cy]
            elif pos == "br": pts_src[3] = [cx, cy]

    # 5. Apply Perspective Transform
    pts_dst = np.array([[0, 0], [WIDTH - 1, 0], [0, HEIGHT - 1], [WIDTH - 1, HEIGHT - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    roi = cv2.warpPerspective(image, M, (WIDTH, HEIGHT))

    # 6. Analyze the Region of Interest (ROI)
    plant_area = find_plant_area(roi)

    # The following analysis is now run on the plant area.
    all_sections = split_area_into_plants(plant_area, BLOCK_ROWS, BLOCK_COLS)

    section_scores = []
    for label, block in all_sections:
        score = get_infection_score(block, HSV_YELLOW_RANGE)
        section_scores.append((label, score))

    block1_results = section_scores[:3]
    block2_results = section_scores[3:]

    # The result now shows which section of the helipad has the most yellow pixels.
    most_infected_b1 = max(block1_results, key=lambda item: item[1])[0]
    most_infected_b2 = max(block2_results, key=lambda item: item[1])[0]

    # 7. Write Final Results to File
    Path("1894.txt").touch(exist_ok=True)
    with open("1894.txt", "w") as f:
        f.write(f"Detected marker IDs: {ids.tolist()}\n")
        f.write(print(f"Infected plant in Block 1: P1{most_infected_b1}\n"))
        f.write(print(f"Infected plant in Block 2: P2{most_infected_b2}\n"))

    return 0


def main():
    '''
    Purpose:
    ---
    This is the main entry point of the script. It parses the command-line
    arguments and calls the main detection function.
    '''
    parser = argparse.ArgumentParser(description="KrishiDrone Area Analyzer")
    parser.add_argument('--image', type=str, required=True,
                        help="Path to the input image for analysis.")
    args = parser.parse_args()

    Detection(args.image)
    sys.exit(0)


if __name__ == "__main__":
    main()
