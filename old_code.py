import cv2
import numpy as np
import cv2.aruco as aruco
import argparse
from pathlib import Path
import sys

# --- Main Configurations ---
WIDTH, HEIGHT = 800, 800
# This HSV range is specifically tuned to identify the bright yellow infection spots.
HSV_YELLOW_RANGE = (np.array([20, 50, 100]), np.array([35, 255, 255]))


# ---------------------- Helper Functions ---------------------- #

def find_and_warp_aruco(image):
    """Detects ArUco markers and performs perspective warping if possible."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1)

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)

    # --- CHANGE 1: Use DetectorParameters_create() for OpenCV 3.x ---
    # The modern `aruco.DetectorParameters()` class is not available in v3.
    parameters = aruco.DetectorParameters_create()

    # Setting parameters remains the same.
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 40
    parameters.adaptiveThreshWinSizeStep = 4
    parameters.adaptiveThreshConstant = 7
    parameters.minMarkerPerimeterRate = 0.02
    parameters.polygonalApproxAccuracyRate = 0.04

    # --- CHANGE 2: Use the direct detectMarkers function ---
    # OpenCV 3.x does not use the `ArucoDetector` class. Detection is a direct function call.
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is None or len(ids) < 4:
        raise RuntimeError("❌ Detection failed: Need at least 4 ArUco markers.")

    centers = np.array([c[0].mean(axis=0) for c in corners])
    sums = centers.sum(axis=1)
    diffs = np.diff(centers, axis=1)

    top_left_idx, bottom_right_idx = np.argmin(sums), np.argmax(sums)
    top_right_idx, bottom_left_idx = np.argmax(diffs), np.argmin(diffs)

    pts_src = np.zeros((4, 2), dtype=np.float32)
    pts_src[0] = corners[top_left_idx][0][0]
    pts_src[1] = corners[top_right_idx][0][1]
    pts_src[2] = corners[bottom_right_idx][0][2]
    pts_src[3] = corners[bottom_left_idx][0][3]

    pts_dst = np.array([[0, 0], [WIDTH - 1, 0], [WIDTH - 1, HEIGHT - 1], [0, HEIGHT - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(image, M, (WIDTH, HEIGHT))

    print(f"✅ ArUco markers detected: {len(ids)} -> IDs: {ids.flatten()}")
    return warped, ids.flatten()


def find_soil_area(half_image):
    """Counts brown/soil pixels to identify the half of the image with the plant area."""
    if half_image.size == 0: return 0
    hsv = cv2.cvtColor(half_image, cv2.COLOR_BGR2HSV)
    lower_brown = np.array([10, 50, 20])
    upper_brown = np.array([30, 255, 200])
    soil_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    return np.sum(soil_mask)


def apply_clahe(image):
    """Applies CLAHE to the L-channel of an LAB image to improve local contrast."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_image


def analyze_region_infection(region_image):
    """
    Splits a region into a 3x2 grid and finds the cell with the largest
    continuous patch of infection.
    """
    h, w, _ = region_image.shape
    block_h, block_w = h // 3, w // 2
    labels = [["A", "D"], ["B", "E"], ["C", "F"]]
    infection_scores = []

    for c in range(2):
        for r in range(3):
            cell = region_image[r * block_h:(r + 1) * block_h, c * block_w:(c + 1) * block_w]
            label = labels[r][c]
            hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, HSV_YELLOW_RANGE[0], HSV_YELLOW_RANGE[1])

            # --- CHANGE 3: Handle 3 return values from findContours ---
            # In OpenCV 3.x, findContours returns (image, contours, hierarchy).
            # We discard the first and last values using underscores.
            _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            score = 0
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                score = cv2.contourArea(largest_contour)

            infection_scores.append((label, score))

    if not infection_scores: return "N/A"
    most_infected_label = max(infection_scores, key=lambda item: item[1])[0]
    return most_infected_label


# ---------------------- Main Detection Function ---------------------- #

def Detection(image_path):
    """
    Main function to run the complete plant infection detection pipeline.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Fatal Error: Could not read image from '{image_path}'.")
        return 1  # Return error code

    try:
        warped_arena, detected_ids = find_and_warp_aruco(image)
    except RuntimeError as e:
        print(e)
        return 1

    warped_arena = cv2.flip(warped_arena, 1)

    mid_point_y = warped_arena.shape[0] // 2
    top_half = warped_arena[:mid_point_y, :]
    bottom_half = warped_arena[mid_point_y:, :]

    if find_soil_area(bottom_half) > find_soil_area(top_half):
        plant_area_half = bottom_half
        print("🌱 Plant trays identified in the BOTTOM half.")
    else:
        plant_area_half = top_half
        print("🌱 Plant trays identified in the TOP half.")

    mid_point_x = plant_area_half.shape[1] // 2
    left_region = plant_area_half[:, :mid_point_x]
    right_region = plant_area_half[:, mid_point_x:]

    if left_region.size == 0 or right_region.size == 0:
        print("❌ Error: Failed to split plant area into two regions.")
        return 1

    left_enhanced = apply_clahe(left_region)
    right_enhanced = apply_clahe(right_region)

    most_infected_b1 = analyze_region_infection(left_enhanced)
    most_infected_b2 = analyze_region_infection(right_enhanced)

    print(f" Infection found in Block 1 at position: P1{most_infected_b1}")
    print(f" Infection found in Block 2 at position: P2{most_infected_b2}")

    # --- Write Results to File ---
    output_filename = "1894.txt"
    # The Path class is modern but available in Python 3.4+, so it's safe to use.
    Path(output_filename).touch(exist_ok=True)
    with open(output_filename, "w") as f:
        f.write(f"Detected marker IDs: {detected_ids.tolist()}\n")
        f.write(f"Infected plant in Block 1: P1{most_infected_b1}\n")
        f.write(f"Infected plant in Block 2: P2{most_infected_b2}\n")

    print(f"✅ Analysis complete. Results saved to {output_filename}")
    return 0


# ---------------------- Entry Point ---------------------- #

def main():
    parser = argparse.ArgumentParser(description="Detect infected plants from a top-down image.")
    parser.add_argument('--image', type=str, required=True, help="Path to the input image file.")
    args = parser.parse_args()

    exit_code = Detection(args.image)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
