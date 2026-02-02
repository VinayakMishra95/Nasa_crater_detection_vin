import os
from pathlib import Path
import csv
import cv2
import numpy as np

# =========================
# CONFIG
# =========================
DATASET_ROOT = Path(r"C:\Users\vinay\nasa-craters-data\test")
OUTPUT_CSV = Path("solution.csv")

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# =========================
# DATASET UTILS
# =========================
def collect_images(dataset_root: Path):
    images = []
    for root, _, files in os.walk(dataset_root):
        for f in files:
            if f.lower().endswith(IMAGE_EXTENSIONS):
                images.append(Path(root) / f)
    return images


def image_id_from_path(img_path: Path, dataset_root: Path):
    """
    Converts:
    C:/.../test/altitude01/longitude05/orientation01_light01.png
    → altitude01/longitude05/orientation01_light01
    """
    rel = img_path.relative_to(dataset_root)
    parts = rel.parts
    image_name = img_path.stem
    return f"{parts[0]}/{parts[1]}/{image_name}"


# =========================
# CRATER DETECTOR
# =========================
def detect_craters(img: np.ndarray):
    """
    Detect craters using edge detection + contour analysis + ellipse fitting.
    Returns list of ellipses: (cx, cy, a, b, angle_deg)
    """
    # Step 1: Preprocessing
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)   # smooth noise
    _, img_thresh = cv2.threshold(img_blur, 30, 255, cv2.THRESH_BINARY)  # simple threshold

    # Step 2: Edge detection
    edges = cv2.Canny(img_thresh, 50, 150)

    # Step 3: Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        if len(cnt) < 5:
            continue  # ellipse fitting requires at least 5 points

        ellipse = cv2.fitEllipse(cnt)
        (cx, cy), (major, minor), angle = ellipse

        # Filter small artifacts
        if major < 10 or minor < 10:
            continue

        # Ensure major ≥ minor
        if minor > major:
            major, minor = minor, major

        detections.append((cx, cy, major / 2, minor / 2, angle))  # OpenCV returns full axis, we want semi

    return detections


# =========================
# MAIN PIPELINE
# =========================
def main():
    print("OpenCV:", cv2.__version__)
    print("NumPy :", np.__version__)
    print("Dataset:", DATASET_ROOT)

    images = collect_images(DATASET_ROOT)
    print(f"Found {len(images)} test images")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f, delimiter=",")

        # HEADER (must match exactly)
        writer.writerow([
            "ellipseCenterX(px)",
            "ellipseCenterY(px)",
            "ellipseSemimajor(px)",
            "ellipseSemiminor(px)",
            "ellipseRotation(deg)",
            "inputImage",
            "crater_classification"
        ])

        for idx, img_path in enumerate(images, 1):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print("Failed to load:", img_path)
                continue

            image_id = image_id_from_path(img_path, DATASET_ROOT)
            detections = detect_craters(img)

            # =========================
            # ZERO-CRATER CASE (MANDATORY)
            # =========================
            if len(detections) == 0:
                writer.writerow([
                    -1, -1, -1, -1, -1,
                    image_id,
                    -1
                ])
            else:
                for (cx, cy, a, b, angle) in detections:
                    writer.writerow([
                        round(float(cx), 2),
                        round(float(cy), 2),
                        round(float(a), 2),
                        round(float(b), 2),
                        round(float(angle), 2),
                        image_id,
                        -1  # classification not used
                    ])

            if idx % 100 == 0:
                print(f"Processed {idx}/{len(images)}")

    print("DONE → solution.csv generated")


# =========================
if __name__ == "__main__":
    main()
