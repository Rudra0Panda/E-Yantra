import cv2
import numpy as np
import cv2.aruco as aruco
import argparse
from pathlib import Path
import sys


class Config:
    """Holds all configuration parameters for the detection pipeline."""
    WARP_WIDTH, WARP_HEIGHT = 800, 800

    # Infection color ranges
    HSV_YELLOW_LOWER = np.array([20, 100, 100])
    HSV_YELLOW_UPPER = np.array([35, 255, 255])
    HSV_INFECTED_BROWN_LOWER = np.array([10, 60, 50])
    HSV_INFECTED_BROWN_UPPER = np.array([30, 255, 200])

    # Soil color range
    HSV_SOIL_BROWN_LOWER = np.array([10, 50, 20])
    HSV_SOIL_BROWN_UPPER = np.array([30, 255, 200])

    # Grey plant tray color range
    HSV_GREY_TRAY_LOWER = np.array([0, 0, 100])
    HSV_GREY_TRAY_UPPER = np.array([180, 50, 220])

    MORPH_KERNEL = np.ones((5, 5), np.uint8)

    # --- CORRECTED LINES for newer OpenCV versions ---
    ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
    ARUCO_PARAMS = aruco.DetectorParameters_create()
    # --- END OF CORRECTION ---

    REQUIRED_IDS = [80, 85, 90, 95]


def find_and_warp_arena(image):
    """Detects ArUco markers and performs perspective correction."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, Config.ARUCO_DICT, parameters=Config.ARUCO_PARAMS)
    if ids is None or len(ids) < 4:
        print("Error: Not all ArUco markers were found.")
        return None, None

    ids = ids.flatten()
    marker_dict = {id_: corner.reshape(4, 2) for id_, corner in zip(ids, corners)}
    if not all(id_ in marker_dict for id_ in Config.REQUIRED_IDS):
        print(f"Error: Missing required ArUco IDs. Found: {ids.tolist()}")
        return None, None

    top_left, top_right = marker_dict[80][0], marker_dict[85][1]
    bottom_right, bottom_left = marker_dict[90][2], marker_dict[95][3]
    src_pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    dst_pts = np.array([[0, 0], [Config.WARP_WIDTH - 1, 0], [Config.WARP_WIDTH - 1, Config.WARP_HEIGHT - 1],
                        [0, Config.WARP_HEIGHT - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, matrix, (Config.WARP_WIDTH, Config.WARP_HEIGHT))
    return warped, ids


def extract_plant_area(warped_image):
    """Dynamically identifies the half of the arena containing the plants."""
    mid_y = warped_image.shape[0] // 2
    top_half, bottom_half = warped_image[:mid_y, :], warped_image[mid_y:, :]
    mask_top = cv2.inRange(cv2.cvtColor(top_half, cv2.COLOR_BGR2HSV), Config.HSV_SOIL_BROWN_LOWER,
                           Config.HSV_SOIL_BROWN_UPPER)
    mask_bottom = cv2.inRange(cv2.cvtColor(bottom_half, cv2.COLOR_BGR2HSV), Config.HSV_SOIL_BROWN_LOWER,
                              Config.HSV_SOIL_BROWN_UPPER)
    ''''''
    if cv2.countNonZero(mask_top) > cv2.countNonZero(mask_bottom):

        return top_half
    else:

        return bottom_half


def analyze_infection(plant_area):
    """Analyzes each plant cell for infection after isolating the plant trays."""
    hsv_area = cv2.cvtColor(plant_area, cv2.COLOR_BGR2HSV)
    tray_mask = cv2.inRange(hsv_area, Config.HSV_GREY_TRAY_LOWER, Config.HSV_GREY_TRAY_UPPER)

    close_kernel = np.ones((15, 15), np.uint8)
    tray_mask = cv2.morphologyEx(tray_mask, cv2.MORPH_CLOSE, close_kernel)

    clean_plant_area = cv2.bitwise_and(plant_area, plant_area, mask=tray_mask)


    h, w, _ = plant_area.shape
    rows, cols = 3, 2
    block_h, block_w = h // rows, w // cols
    labels = [["A", "D"], ["B", "E"], ["C", "F"]]
    cell_scores = []
    overlay = plant_area.copy()

    for c in range(cols):
        for r in range(rows):
            x1, y1 = c * block_w, r * block_h
            x2, y2 = (c + 1) * block_w, (r + 1) * block_h

            cell = clean_plant_area[y1:y2, x1:x2]

            blurred_cell = cv2.GaussianBlur(cell, (7, 7), 0)
            hsv_cell = cv2.cvtColor(blurred_cell, cv2.COLOR_BGR2HSV)
            mask_yellow = cv2.inRange(hsv_cell, Config.HSV_YELLOW_LOWER, Config.HSV_YELLOW_UPPER)
            mask_brown = cv2.inRange(hsv_cell, Config.HSV_INFECTED_BROWN_LOWER, Config.HSV_INFECTED_BROWN_UPPER)
            infection_mask = cv2.bitwise_or(mask_yellow, mask_brown)
            infection_mask = cv2.morphologyEx(infection_mask, cv2.MORPH_OPEN, Config.MORPH_KERNEL)
            score = cv2.countNonZero(infection_mask)
            cell_scores.append({"label": labels[r][c], "score": score})

            label = labels[r][c]
            index = r + c * rows + 1
            color = (0, 0, 255) if score > 100 else (0, 255, 0)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
            cv2.putText(overlay, f"{label} (#{index})", (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(overlay, f"Score: {score}", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)



    block1 = [cell for cell in cell_scores if cell["label"] in ["A", "B", "C"]]
    block2 = [cell for cell in cell_scores if cell["label"] in ["D", "E", "F"]]
    most_infected_b1 = max(block1, key=lambda x: x["score"])['label']
    most_infected_b2 = max(block2, key=lambda x: x["score"])['label']

    return most_infected_b1, most_infected_b2


def Detection(image_path):
    """Main function to run the entire detection pipeline."""
    print(f"Starting analysis for image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"FATAL: Image not found at {image_path}")
        return 1

    warped_arena, detected_ids = find_and_warp_arena(image)
    if warped_arena is None:
        return 1



    plant_area = extract_plant_area(warped_arena)


    infected_plant_1, infected_plant_2 = analyze_infection(plant_area)

    print("-" * 30)
    print(f"Infection found in Block 1 at position: P1{infected_plant_1}")
    print(f"Infection found in Block 2 at position: P2{infected_plant_2}")
    print("-" * 30)

    final_image = cv2.imread("plant_tray_analysis.png")
    #cv2.imshow("Final Analysis", final_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    input_path = Path(image_path)
    output_txt_path = input_path.with_name("1894.txt")

    print(f"Saving results to: {output_txt_path}")

    with open(output_txt_path, "w") as f:
        f.write(f"Detected marker IDs: {detected_ids.tolist() if detected_ids is not None else 'None'}\n")
        f.write(f"Infected plant in Block 1: P1{infected_plant_1}\n")
        f.write(f"Infected plant in Block 2: P2{infected_plant_2}\n")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Detect infected plants with visual labels")
    parser.add_argument('--image', type=str, required=True, help="Path to input image")
    args = parser.parse_args()
    sys.exit(Detection(args.image))


if __name__ == "__main__":
    main()
