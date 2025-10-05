'''
# Team ID:          eYRC#1894
# Theme:            KrishiDrone
# Author List:      1.Rudra Narayan Panda
#                   2.Mahasangram Kar
#                   3.Sidhartha Kumar Nayak
#                   4.Subham Kumar Rana
# Filename:         task1a.py
# Functions:        split_area_into_plants, get_infection_score, main
# Global variables: ID_TO_POS, WIDTH, HEIGHT, BLOCK_ROWS, BLOCK_COLS, HSV_YELLOW_RANGE
'''

import cv2
import numpy as np
import cv2.aruco as aruco

# -- Configurations --
# ID_TO_POS: Dictionary mapping specific ArUco marker IDs to their physical corner positions.
ID_TO_POS = {85: "tl", 90: "tr", 80: "bl", 95: "br"}
# WIDTH, HEIGHT: The desired dimensions (width, height) of the final top-down image.
WIDTH, HEIGHT = 640, 500
# BLOCK_ROWS, BLOCK_COLS: The grid size (rows, columns) for analyzing the plant area.
BLOCK_ROWS, BLOCK_COLS = 3, 2
# HSV_YELLOW_RANGE: The lower and upper HSV color values to detect yellow (infection).
HSV_YELLOW_RANGE = (np.array([20, 100, 100]), np.array([35, 255, 255]))


def split_area_into_plants(area, rows=3, cols=2):
    '''
    Purpose:
    ---
    Splits a given image area into a grid of smaller blocks and assigns a
    pre-defined label to each block.
    '''
    # Get the height and width of the input area
    h, w = area.shape[:2]
    # Calculate the height and width of a single block
    block_h, block_w = h // rows, w // cols
    # Define the labels for each position in the 3x2 grid
    labels = [["A", "D"], ["B", "E"], ["C", "F"]]
    blocks = []

    # Iterate through columns first, then rows, to group plants from the
    # first column (A,B,C) before moving to the second column (D,E,F).
    for c in range(cols):
        for r in range(rows):
            # Crop the image to get the current block
            block = area[r * block_h:(r + 1) * block_h, c * block_w:(c + 1) * block_w]
            # Get the corresponding label
            label = labels[r][c]
            # Append the label and the block image to the list
            blocks.append((label, block))
    return blocks


def get_infection_score(block, hsv_range):
    '''
    Purpose:
    ---
    Calculates the number of "infected" pixels in an image block.
    '''
    # Convert the block from BGR color space to HSV color space
    hsv = cv2.cvtColor(block, cv2.COLOR_BGR2HSV)
    # Get the lower and upper bounds of the target color
    lower, upper = hsv_range
    # Create a mask where pixels within the HSV range are white, and others are black
    mask = cv2.inRange(hsv, lower, upper)
    # Count the number of white pixels in the mask to get the score
    return cv2.countNonZero(mask)


def main():
    '''
    Purpose:
    ---
    This is the main function that executes the entire pipeline.
    '''
    # 1. Load and Prepare the Image
    # ---------------------------------
    # Load the image from the specified file path
    image = cv2.imread("task1a_image.jpg")
    if image is None:
        raise FileNotFoundError("Image not found. Check path!")

    # Resize for consistent processing and convert to grayscale for ArUco detection
    image = cv2.resize(image, (640, 480))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Enhance Image for Better Detection
    # ---------------------------------------
    # Start with the clean grayscale image.
    processed_gray = gray

     ## Optional: To handle shadows
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    processed_gray = clahe.apply(processed_gray)

    # 3. Detect ArUco Markers
    # -------------------------
    # Define the ArUco dictionary and get default parameters
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
    parameters = aruco.DetectorParameters()
    # Enable sub-pixel corner refinement for better accuracy on angled/blurry markers
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    # Create the detector with the specified dictionary and tuned parameters
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    # Find all ArUco markers in the (potentially enhanced) grayscale image
    corners, ids, _ = detector.detectMarkers(processed_gray)

    # Check if any markers were detected and raise an error if none were found
    if ids is None:
        raise RuntimeError("No ArUco markers detected!")

    # Simplify the IDs array from a 2D array to a 1D list
    ids = ids.flatten()
    print(f"Detected marker IDs: {ids.tolist()}")

    # 4. Map Source and Destination Points
    # --------------------------------------
    # Prepare an empty array to hold the four source points for the transform
    pts_src = np.zeros((4, 2), dtype="float32")
    # Loop through the detected markers and map their IDs to the correct corner position
    for corner, marker_id in zip(corners, ids):
        c = corner[0]
        # Calculate the center (cx, cy) of the marker
        cx, cy = np.mean(c, axis=0)
        # Check if the detected marker ID is one of our four corner markers
        if marker_id in ID_TO_POS:
            pos = ID_TO_POS[marker_id]
            # Place the center coordinate into the correct slot in the pts_src array
            if pos == "tl":
                pts_src[0] = [cx, cy]
            elif pos == "tr":
                pts_src[1] = [cx, cy]
            elif pos == "bl":
                pts_src[2] = [cx, cy]
            elif pos == "br":
                pts_src[3] = [cx, cy]

    # Define the four corners of the new, rectangular output image
    pts_dst = np.array([[0, 0], [WIDTH - 1, 0], [0, HEIGHT - 1], [WIDTH - 1, HEIGHT - 1]], dtype="float32")

    # 5. Apply Perspective Transform
    # --------------------------------
    # Calculate the transformation matrix 'M' based on the source and destination points
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    # Apply the transformation to the original image to get the flat, top-down view (ROI)
    roi = cv2.warpPerspective(image, M, (WIDTH, HEIGHT))

    # 6. Analyze the Region of Interest (ROI)
    # -----------------------------------------
    # Crop the right half of the ROI, which contains the plant beds
    roi_h, roi_w = roi.shape[:2]
    plant_area = roi[:, roi_w // 2:]

    # Split the plant area into 6 individual plant blocks (A-F)
    all_plants = split_area_into_plants(plant_area, BLOCK_ROWS, BLOCK_COLS)

    # Calculate an infection score for each plant block
    infection_scores = []
    for label, block in all_plants:
        score = get_infection_score(block, HSV_YELLOW_RANGE)
        infection_scores.append((label, score))

    # Group the results into two blocks (left and right columns)
    # Block 1 corresponds to the first 3 plants in the list (A, B, C)
    block1_results = infection_scores[:3]
    # Block 2 corresponds to the next 3 plants (D, E, F)
    block2_results = infection_scores[3:]

    # Find the plant with the highest infection score in each block
    # The 'key' tells the max() function to compare items based on the score (the second element)
    most_infected_b1 = max(block1_results, key=lambda item: item[1])[0]
    most_infected_b2 = max(block2_results, key=lambda item: item[1])[0]

    # 7. Print the Final Results
    # --------------------------
    print(f"Infected plant in Block 1: P1{most_infected_b1}")
    print(f"Infected plant in Block 2: P2{most_infected_b2}")


# This is the standard entry point that calls the main() function when the script is executed
if __name__ == "__main__":
    main()
