import cv2
import numpy as np
import cv2.aruco as aruco
import argparse
from pathlib import Path
import sys


def Detection(image_path):
    """
    Main detection function that encapsulates all logic: loading, warping,
    enhancing, filtering, analyzing, and writing the result to a file.
    """

    # --- All configurations and helper functions are now nested inside ---

    # A centralized class to hold all tunable parameters.
    class Config:
        WIDTH, HEIGHT = 800, 800
        HSV_YELLOW_LOWER = np.array([20, 50, 100])
        HSV_YELLOW_UPPER = np.array([35, 255, 255])
        HSV_BROWN_LOWER = np.array([10, 50, 20])
        HSV_BROWN_UPPER = np.array([30, 255, 200])
        ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
        ARUCO_PARAMS = aruco.DetectorParameters_create()
        ARUCO_PARAMS.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        ARUCO_PARAMS.adaptiveThreshWinSizeMin = 3
        ARUCO_PARAMS.adaptiveThreshWinSizeMax = 23
        ARUCO_PARAMS.adaptiveThreshWinSizeStep = 2
        ARUCO_PARAMS.adaptiveThreshConstant = 5
        CLAHE_CLIP_LIMIT = 2.0
        CLAHE_GRID_SIZE = (8, 8)
        MORPH_KERNEL = np.ones((3, 3), np.uint8)
        MORPH_ITERATIONS = 2

    # Helper function to find and warp the image based on ArUco markers.
    def find_and_warp_aruco(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=Config.CLAHE_CLIP_LIMIT, tileGridSize=Config.CLAHE_GRID_SIZE)
        gray = clahe.apply(gray)
        corners, ids, _ = aruco.detectMarkers(gray, Config.ARUCO_DICT, parameters=Config.ARUCO_PARAMS)

        if ids is None or len(ids) < 4:
            # Return None if detection fails, to be handled by the main logic
            return None, None

        centers = np.array([c[0].mean(axis=0) for c in corners])
        sums = centers.sum(axis=1)
        diffs = np.diff(centers, axis=1)
        top_left_idx, bottom_right_idx = np.argmin(sums), np.argmax(sums)
        top_right_idx, bottom_left_idx = np.argmax(diffs), np.argmin(diffs)

        pts_src = np.array([
            corners[top_left_idx][0][0], corners[top_right_idx][0][1],
            corners[bottom_right_idx][0][2], corners[bottom_left_idx][0][3]
        ], dtype=np.float32)

        pts_dst = np.array([
            [0, 0], [Config.WIDTH - 1, 0],
            [Config.WIDTH - 1, Config.HEIGHT - 1], [0, Config.HEIGHT - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped = cv2.warpPerspective(image, M, (Config.WIDTH, Config.HEIGHT))
        return warped, ids

    # Helper function to count soil pixels.
    def find_soil_area(half_image):
        if half_image.size == 0: return 0
        hsv = cv2.cvtColor(half_image, cv2.COLOR_BGR2HSV)
        soil_mask = cv2.inRange(hsv, Config.HSV_BROWN_LOWER, Config.HSV_BROWN_UPPER)
        return np.sum(soil_mask)

    # Helper function for contrast enhancement.
    def apply_clahe(image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=Config.CLAHE_CLIP_LIMIT, tileGridSize=Config.CLAHE_GRID_SIZE)
        cl = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    # Helper function to analyze each region for infection.
    def analyze_region_infection(region_image):
        h, w, _ = region_image.shape
        block_h, block_w = h // 3, w // 2
        labels = [["A", "D"], ["B", "E"], ["C", "F"]]
        cell_analysis_results = []

        for c in range(2):
            for r in range(3):
                x, y = c * block_w, r * block_h
                cell = region_image[y:y + block_h, x:x + block_w]
                hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, Config.HSV_YELLOW_LOWER, Config.HSV_YELLOW_UPPER)
                mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, Config.MORPH_KERNEL,
                                                iterations=Config.MORPH_ITERATIONS)
                score = cv2.countNonZero(mask_cleaned)
                cell_analysis_results.append({"label": labels[r][c], "score": score})

        if not cell_analysis_results:
            return "N/A"

        most_infected_cell = max(cell_analysis_results, key=lambda x: x["score"])
        return most_infected_cell['label']

    # --- Main Pipeline Logic ---
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not read image from '{image_path}'.")
        return 1

    warped_arena, ids = find_and_warp_aruco(image)
    if warped_arena is None:
        print("❌ Detection failed: Need at least 4 ArUco markers.")
        return 1

    print(f"✅ ArUco markers detected: {len(ids)} -> IDs: {ids.flatten()}")
    warped_arena = cv2.flip(warped_arena, 1)

    mid_y = warped_arena.shape[0] // 2
    top_half, bottom_half = warped_arena[:mid_y, :], warped_arena[mid_y:, :]

    if find_soil_area(bottom_half) > find_soil_area(top_half):
        plant_area_half = bottom_half
        print("🌱 Plant trays identified in the BOTTOM half.")
    else:
        plant_area_half = top_half
        print("🌱 Plant trays identified in the TOP half.")

    mid_x = plant_area_half.shape[1] // 2
    left_region, right_region = plant_area_half[:, :mid_x], plant_area_half[:, mid_x:]

    print("🔬 Enhancing image contrast and analyzing for infection...")
    left_enhanced = apply_clahe(left_region)
    right_enhanced = apply_clahe(right_region)

    most_infected_b1 = analyze_region_infection(left_enhanced)
    most_infected_b2 = analyze_region_infection(right_enhanced)

    print(f"📊 Infection found in Block 1 at position: P1{most_infected_b1}")
    print(f"📊 Infection found in Block 2 at position: P2{most_infected_b2}")

    # --- Write Results to File ---
    print("✅ Analysis complete. Writing results to 1894.txt...")
    Path("1894.txt").touch(exist_ok=True)
    with open("1894.txt", "w") as f:
        f.write(f"Detected marker IDs: {ids.flatten().tolist()}\n")
        f.write(f"Infected plant in Block 1: P1{most_infected_b1}\n")
        f.write(f"Infected plant in Block 2: P2{most_infected_b2}\n")

    return 0


def main():
    """Parses command-line arguments and calls the main detection function."""
    parser = argparse.ArgumentParser(description="Detect infected plants and write results to a file.")
    parser.add_argument('--image', type=str, required=True, help="Path to the input image file.")
    args = parser.parse_args()

    # Exit with the return code from the Detection function
    sys.exit(Detection(args.image))


if __name__ == "__main__":
    main()
