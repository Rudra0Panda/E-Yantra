import cv2
import numpy as np
import cv2.aruco as aruco
import argparse
from pathlib import Path
import sys

# --- Main Configurations ---
WIDTH, HEIGHT = 600, 500
TRAY_MIN_AREA = 7000
TRAY_MAX_AREA = 35000
# This HSV range is specifically tuned to identify the bright yellow infection spots.
HSV_YELLOW_RANGE = (np.array([22, 90, 90]), np.array([38, 255, 255]))


# ---------------------- Core Helper Functions ---------------------- #

def enhance_image(image):
    """Enhances image contrast and sharpens details for better detection."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    return enhanced


def get_aruco_corners_precise(gray):
    """Dynamically detects ArUco markers and sorts their corners for perspective transform."""
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) < 4:
        raise RuntimeError("Detection failed: Need at least 4 ArUco markers!")

    centers = np.array([np.mean(c[0], axis=0) for c in corners])
    sums = centers[:, 0] + centers[:, 1]
    diffs = centers[:, 0] - centers[:, 1]

    pts_src = np.zeros((4, 2), dtype=np.float32)
    pts_src[0] = centers[np.argmin(sums)]  # Top-Left
    pts_src[1] = centers[np.argmax(diffs)]  # Top-Right
    pts_src[3] = centers[np.argmax(sums)]  # Bottom-Right
    pts_src[2] = centers[np.argmin(diffs)]  # Bottom-Left

    return pts_src, ids.flatten()


def is_plant_area(half_image):
    """Counts white pixels to identify the half of the image containing the plant trays."""
    if half_image.size == 0: return 0
    gray = cv2.cvtColor(half_image, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    return np.sum(white_mask // 255)


def extract_grouped_regions(plant_area):
    """Dynamically finds the two tray groups (regions) within the plant area using contours."""
    gray = cv2.cvtColor(plant_area, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 10))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    region_boxes = []

    min_region_area = 1.5 * TRAY_MIN_AREA
    max_region_area = 3.0 * TRAY_MAX_AREA

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_region_area < area < max_region_area:
            region_boxes.append(cv2.boundingRect(cnt))

    if len(region_boxes) != 2:
        return None, None

    region_boxes.sort(key=lambda b: b[0])
    x1, y1, w1, h1 = region_boxes[0]
    region1 = plant_area[y1:y1 + h1, x1:x1 + w1]
    x2, y2, w2, h2 = region_boxes[1]
    region2 = plant_area[y2:y2 + h2, x2:x2 + w2]
    return region1, region2


def analyze_region_infection(region_image):
    """Splits a region into a 3x2 grid and finds the most infected cell."""
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
            score = cv2.countNonZero(mask)
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
        print(f"Fatal Error: Could not read image from '{image_path}'.")
        return 1  # Return error code

    # --- Pipeline Steps ---
    image = cv2.resize(image, (WIDTH, HEIGHT))
    image = enhance_image(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    try:
        pts_src, detected_ids = get_aruco_corners_precise(gray)
    except RuntimeError as e:
        print(e)
        return 1

    pts_dst = np.array([[0, 0], [WIDTH - 1, 0], [0, HEIGHT - 1], [WIDTH - 1, HEIGHT - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    roi = cv2.warpPerspective(image, M, (WIDTH, HEIGHT))

    H, W, _ = roi.shape
    candidates = {"bottom": roi[H // 2:, :], "top": roi[:H // 2, :], "left": roi[:, :W // 2], "right": roi[:, W // 2:]}
    white_counts = {key: is_plant_area(img) for key, img in candidates.items()}
    best_candidate_key = max(white_counts, key=white_counts.get)
    plant_area = candidates[best_candidate_key]

    if best_candidate_key == "left":
        plant_area = cv2.rotate(plant_area, cv2.ROTATE_90_CLOCKWISE)
    elif best_candidate_key == "right":
        plant_area = cv2.rotate(plant_area, cv2.ROTATE_90_COUNTERCLOCKWISE)
    plant_area = cv2.rotate(plant_area, cv2.ROTATE_180)

    block1_region, block2_region = extract_grouped_regions(plant_area)

    if block1_region is None or block2_region is None:
        print("Error: Could not extract the two plant blocks.")
        return 1

    most_infected_b1 = analyze_region_infection(block1_region)
    most_infected_b2 = analyze_region_infection(block2_region)

    # --- Write Results to File ---
    output_filename = "1894.txt"
    Path(output_filename).touch(exist_ok=True)
    with open(output_filename, "w") as f:
        f.write(f"Detected marker IDs: {detected_ids.tolist()}\n")
        f.write(f"Infected plant in Block 1: P1{most_infected_b1}\n")
        f.write(f"Infected plant in Block 2: P2{most_infected_b2}\n")

    print(f"Analysis complete. Results saved to {output_filename}")
    return 0


# ---------------------- Entry Point ---------------------- #

def main():
    parser = argparse.ArgumentParser(description="Detect infected plants from an image.")
    parser.add_argument('--image', type=str, required=True, help="Path to the input image file.")
    args = parser.parse_args()

    # Run the detection and get the exit code
    exit_code = Detection(args.image)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
