import cv2
import numpy as np
import cv2.aruco as aruco
import argparse
from pathlib import Path
import sys

def Detection(image_path):




        
    Path("1894.txt").touch(exist_ok=True)
    with open("1894.txt", "w") as f:
        f.write(f"Detected marker IDs: {ids.tolist()}\n")
        f.write(f"Infected plant in Block 1: P1{most_infected_b1}\n")
        f.write(f"Infected plant in Block 2: P2{most_infected_b2}\n")

    return 0  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    args = parser.parse_args()
    
    Detection(args.image)
    sys.exit(0)  

if __name__ == "__main__":
    main()
