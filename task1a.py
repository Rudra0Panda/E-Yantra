'''
# Team ID:          eYRC#1894
# Theme:            KrishiDrone
# Author List:      1.Rudra Narayan Panda
                    2.Mahasangram Kar
                    3.Sidhartha Kumar Nayak
                    4.Subham Kumar Rana
# Filename:         task1a
# Functions:        split_area_into_plants, get_infection_score, main
# Global variables: ID_TO_POS, WIDTH, HEIGHT, BLOCK_ROWS, BLOCK_COLS, HSV_YELLOW_RANGE
'''

import cv2
import numpy as np
import cv2.aruco as aruco

# -- Configurations --
# ID_TO_POS: Dictionary mapping ArUco marker IDs to their corner positions (tl: top-left).
ID_TO_POS = {85: "tl", 90: "tr", 80: "bl", 95: "br"}
# WIDTH, HEIGHT: Dimensions of the Region of Interest (ROI) after perspective transform.
WIDTH, HEIGHT = 640, 500
# BLOCK_ROWS, BLOCK_COLS: Defines the grid layout for the plant area (3 rows, 2 columns).
BLOCK_ROWS, BLOCK_COLS = 3, 2
# HSV_YELLOW_RANGE: Lower and upper bounds for detecting yellow/brown color (infection).
HSV_YELLOW_RANGE = (np.array([20, 100, 100]), np.array([35, 255, 255]))


def split_area_into_plants(area, rows=3, cols=2):
    '''
    Purpose:
    ---
    Splits a given image area into a grid of smaller blocks and assigns a
    pre-defined label to each block.

    Input Arguments:
    ---
    `area` :  [ numpy.ndarray ]
        The input image to be split into a grid.
    `rows` :  [ int ]
        The number of rows to split the image into.
    `cols` :  [ int ]
        The number of columns to split the image into.

    Returns:
    ---
    `blocks` :  [ list ]
        A list of tuples, where each tuple contains a plant label (e.g., 'A')
        and the corresponding image block as a numpy array.

    Example call:
    ---
    plant_blocks = split_area_into_plants(plant_area_image, 3, 2)
    '''
    h, w = area.shape[:2]
    block_h, block_w = h // rows, w // cols
    labels = [["A", "D"], ["B", "E"], ["C", "F"]]
    blocks = []

    # Iterate through columns first, then rows, to group all plants from
    # the first column (A,B,C) before moving to the second column (D,E,F).
    for c in range(cols):
        for r in range(rows):
            block = area[r*block_h:(r+1)*block_h, c*block_w:(c+1)*block_w]
            label = labels[r][c]
            blocks.append((label, block))
    return blocks

def get_infection_score(block, hsv_range):
    '''
    Purpose:
    ---
    Calculates the number of "infected" pixels in an image block by
    counting pixels that fall within a specified HSV color range.

    Input Arguments:
    ---
    `block` :  [ numpy.ndarray ]
        The image of a single plant to be analyzed.
    `hsv_range` :  [ tuple ]
        A tuple containing two numpy arrays for the lower and upper HSV bounds.

    Returns:
    ---
    `infected_pixel_count` :  [ int ]
        The total count of pixels matching the infection color criteria.

    Example call:
    ---
    score = get_infection_score(plant_image, (lower_hsv, upper_hsv))
    '''
    hsv = cv2.cvtColor(block, cv2.COLOR_BGR2HSV)
    lower, upper = hsv_range
    mask = cv2.inRange(hsv, lower, upper)
    return cv2.countNonZero(mask)

def main():
    '''
    Purpose:
    ---
    This is the main function that executes the entire pipeline:
    1. Loads the image.
    2. Detects ArUco markers to define an arena.
    3. Performs a perspective transform to get a top-down view.
    4. Crops the relevant plant area.
    5. Splits the area into individual plants and analyzes each for infection.
    6. Prints the most infected plant for each block.

    Input Arguments:
    ---
    None

    Returns:
    ---
    None

    Example call:
    ---
    main()
    '''
    # Load the image from file
    image = cv2.imread("task1a_image.jpg")
    if image is None:
        raise FileNotFoundError("Image not found. Check path!")

    image = cv2.resize(image, (640, 480))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize ArUco detector and find markers in the image
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        raise RuntimeError("No ArUco markers detected!")

    ids = ids.flatten()
    print(f"Detected marker IDs: {ids.tolist()}")

    # Map the center of each detected ArUco marker to its predefined position
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

    # Apply a perspective transform to get a top-down view of the arena
    pts_dst = np.array([[0, 0], [WIDTH-1, 0], [0, HEIGHT-1], [WIDTH-1, HEIGHT-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    roi = cv2.warpPerspective(image, M, (WIDTH, HEIGHT))

    # Crop the right half of the transformed image, which contains the plant beds
    roi_h, roi_w = roi.shape[:2]
    plant_area = roi[:, roi_w//2:]

    # Analyze the cropped plant area
    all_plants = split_area_into_plants(plant_area, BLOCK_ROWS, BLOCK_COLS)

    infection_scores = []
    for label, block in all_plants:
        score = get_infection_score(block, HSV_YELLOW_RANGE)
        infection_scores.append((label, score))

    # The first 3 items in the list correspond to Block 1 (plants A, B, C)
    block1_results = infection_scores[:3]
    # The next 3 items correspond to Block 2 (plants D, E, F)
    block2_results = infection_scores[3:]

    # Find the plant with the highest infection score in each block.
    # The `key` argument tells max() to look at the score (the second item in the tuple).
    # The `[0]` at the end selects the label (the first item) from the resulting tuple.
    most_infected_b1 = max(block1_results, key=lambda item: item[1])[0]
    most_infected_b2 = max(block2_results, key=lambda item: item[1])[0]

    # Print the final results to the console
    print(f"Infected plant in Block 1: P1{most_infected_b1}")
    print(f"Infected plant in Block 2: P2{most_infected_b2}")

if __name__ == "__main__":
    main()
