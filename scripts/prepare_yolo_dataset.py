#!/usr/bin/env python3

from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from tqdm import tqdm
import rasterio
from rasterio.windows import Window
import numpy as np
from PIL import Image

project_root = str(Path(__file__).resolve().parents[1])
sys.path.append(project_root)
from configs.paths import *


# ================================
# CHIPS CONFIG
# ================================
TILE_SIZE = 640
STRIDE = 320


# ================================
# HELPER FUNCTIONS
# ================================
def parse_xml(xml_path):
    """
    This function reads a .xml file and returns a list of lists. Each inner list contains information of a bbox: xmin, ymin, xmax, ymax
    :param xml_path: Path
    :return: List[List[]]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    bboxes = []

    for bbox in root.iter("bndbox"):
        xmin = bbox.find("xmin")
        ymin = bbox.find("ymin")
        xmax = bbox.find("xmax")
        ymax = bbox.find("ymax")

        if xmin is None or ymin is None or xmax is None or ymax is None:
            raise ValueError(f"Missing box coordinate in {xml_path}")

        info = [
            int(xmin.text),
            int(ymin.text),
            int(xmax.text),
            int(ymax.text)
        ]

        bboxes.append(info)

    return bboxes

def clip_box_2_tile(bbox: list, tile_x, tile_y, tile_size):
    """
    This function is used when chipping the image. The purpose of this function is to clip the original box's coordinate relative to the tile
    :param bbox: list
    :param tile_x: int
    :param tile_y: int
    :param tile_size: int
    :return: list
    """
    xmin, ymin, xmax, ymax = bbox

    # Calculate the new coordinate relative to the tile
    xmin -= tile_x
    ymin -= tile_y
    xmax -= tile_x
    ymax -= tile_y

    # Check if the bbox is in the tile
    if xmin <= 0 or ymin <= 0 or xmax >= tile_size or ymax >= tile_size:
        return None

    return [xmin, ymin, xmax, ymax]


# ================================
# MAIN
# ================================
def main():
    tile_metadata = {}  # {tile_id: {transform, crs, window}}
    tile_count = 1
    for img_path in tqdm(list(RASTER_DIR.glob("*.tif"))):
        img_id = int(img_path.stem)
        xml_path = ANNOTATION_DIR / f"{img_id}.xml"

        src = rasterio.open(img_path, "r")
        img_width, img_height = src.width, src.height
        crs = str(src.crs)

        # Left-right, top-down
        for y in range(0, img_height, STRIDE):
            for x in range(0, img_width, STRIDE):
                # Define a window in the src, then create a tile (which is a new .tif file)
                window = Window(x, y, TILE_SIZE, TILE_SIZE)
                tile = src.read(window=window, boundless=True, fill_value=0)

                # Create train:val:test data from images 1-5:6:7-8
                split = None
                if img_id <= 5:
                    split = "train"
                elif img_id == 6:
                    split = "val"
                else:
                    split = "test"

                # Save the tile

                # tile.shape = (4, 640, 640)
                # Band 1: Red
                # Band 2: Green
                # Band 3: Blue
                # Band 4: NIR (Near-Infrared)
                rgb = tile[[0, 1, 2], :, :].astype(np.float32)  # take the RGB bands only
                rgb = rgb.astype(np.uint8).transpose(1, 2, 0)  # reshape from Channel, Width, Height to Width, Height, Channel
                rgb_img = Image.fromarray(rgb)  # create an image from an array
                rgb_img.save(IMG_DIR / split / f"{tile_count}.png")

                # Save geo metadata in .geojson format
                chip_transform = rasterio.windows.transform(window, src.transform)
                tile_metadata[tile_count] = {
                    "source_image": img_id,
                    "crs": crs,
                    "col_off": x,
                    "row_off": y,
                    "transform": list(chip_transform)[:6]
                }

                # Create a label file for each tile
                label_file = f"{tile_count}.txt"
                label_path = LABELS_DIR / split / label_file

                # Get all the bboxes inside the tile
                img_bboxes = parse_xml(xml_path)
                yolo_lines = []
                for bbox in img_bboxes:
                    clipped = clip_box_2_tile(bbox, x, y, TILE_SIZE)
                    if clipped:
                        xmin, ymin, xmax, ymax = clipped

                        # Calculate and normalized
                        x_center = (xmin + xmax) / 2 / TILE_SIZE
                        y_center = (ymin + ymax) / 2 / TILE_SIZE
                        bbox_width = (xmax - xmin) / TILE_SIZE
                        bbox_height = (ymax - ymin) / TILE_SIZE

                        yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

                with open(label_path, "w") as f:
                    f.write(f"\n".join(yolo_lines))

                tile_count += 1

    print(f"✅ Tiling completed! Dataset saved at: {YOLO_DATASET_DIR}")


if __name__ == "__main__":
    main()