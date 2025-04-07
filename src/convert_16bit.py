import cv2
import numpy as np
import os
import glob

# Paths
input_folder = "/segment_project/data/test/images"
output_folder = "/segment_project/data/test/images_8bit"
os.makedirs(output_folder, exist_ok=True)

# Get all TIFF files
tiff_files = glob.glob(os.path.join(input_folder, "*.tif"))

# Initialize CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))

for file in tiff_files:
    print(f"Loading {file}...")
    
    # Load image safely
    image_16bit = cv2.imread(file, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    
    if image_16bit is None:
        print(f"Error: Could not load {file}. Skipping...")
        continue

    print(f"Processing {file} - Shape: {image_16bit.shape}, Dtype: {image_16bit.dtype}")

    # Resize large images
    max_size = 4096
    h, w = image_16bit.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image_16bit = cv2.resize(image_16bit, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        print(f"Resized {file} to {image_16bit.shape[:2]}")

    # Check number of channels (RGBA)
    if image_16bit.shape[-1] == 4:
        b16, g16, r16, a16 = cv2.split(image_16bit)
        a8 = cv2.normalize(a16, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # Preserve alpha
    else:
        b16, g16, r16 = cv2.split(image_16bit)
        a8 = None

    # Normalize and apply CLAHE
    b8 = clahe.apply(cv2.normalize(b16, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U))
    g8 = clahe.apply(cv2.normalize(g16, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U))
    r8 = clahe.apply(cv2.normalize(r16, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U))

    # Merge channels
    if a8 is not None:
        image_clahe = cv2.merge([b8, g8, r8, a8])
    else:
        image_clahe = cv2.merge([b8, g8, r8])

    # Save output
    output_path = os.path.join(output_folder, os.path.basename(file))
    cv2.imwrite(output_path, image_clahe)

    print(f"Saved: {output_path}")

print("Processing complete!")
