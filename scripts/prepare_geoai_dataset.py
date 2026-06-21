#!/usr/bin/env python3

from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from tqdm import tqdm
import rasterio
from rasterio.windows import Window
import numpy as np
from PIL import Image
import geoai
import geopandas as gpd
from shapely.geometry import Polygon
import shutil

project_root = str(Path(__file__).resolve().parents[1])
sys.path.append(project_root)
from configs.paths import *


# ================================
# CHIPS CONFIG
# ================================
TILE_SIZE = 512
STRIDE = 256


# ================================
# HELPER FUNCTIONS
# ================================
def parse_xml(xml_path):
    """
    This function reads a .xml file and returns a list of dictionaries. Each dictionary is an object containing information of a bbox: xmin, ymin, xmax, ymax
    :param xml_path: Path
    :return: list[dict]
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

        info = {
            "xmin": int(xmin.text),
            "ymin": int(ymin.text),
            "xmax": int(xmax.text),
            "ymax": int(ymax.text)
        }

        bboxes.append(info)

    return bboxes


# ================================
# MAIN
# ================================
def main():
    # Convert the .xml files to .json and put them in the vectors dir
    for xml in ANNOTATION_DIR.iterdir():
        xml_path = str(xml.resolve())
        if ".txt" in xml_path:
            continue

        xml_name = xml.name[0]
        raster_path = RASTER_DIR / f"{xml_name}.tif"

        with rasterio.open(raster_path) as src:
            raster_crs = src.crs
            t = src.transform

        polygons = []
        for bbox in parse_xml(xml_path):
            xmin, ymin = bbox["xmin"], bbox["ymin"]
            xmax, ymax = bbox["xmax"], bbox["ymax"]

            A = t * (xmin, ymin)  # top-left
            B = t * (xmax, ymin)  # top-right
            C = t * (xmax, ymax)  # bottom-right
            D = t * (xmin, ymax)  # bottom-left

            polygons.append(Polygon([A, B, C, D, A]))

        gdf = gpd.GeoDataFrame(
            {"class": ["riceseedling"] * len(polygons)},
            geometry=polygons,
            crs=raster_crs
        )
        out_path = VECTORS_DIR / f"{xml_name}.geojson"
        gdf.to_file(out_path, driver="GeoJSON")
        print(f"{xml_name}: {len(polygons)} annotations, CRS={raster_crs}")

    # Chip down the rasters before feeding to the model
    # Create the temp directories
    TEMP_DIR = GEOAI_DATASET_DIR / "temp"
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    for raster_path in sorted(RASTER_DIR.glob("*.tif")):
        name = raster_path.stem
        vector_path = VECTORS_DIR / f"{name}.geojson"
        print(f"Chipping:\n + raster {raster_path}\n + vector {vector_path}")

        geoai.export_geotiff_tiles(
            in_raster=raster_path,
            out_folder=TEMP_DIR / name,
            in_class_data=vector_path,
            tile_size=TILE_SIZE,
            stride=STRIDE,
            buffer_radius=0,
        )

    # Create train/val data from images 1-6
    train_val_count = 0
    for i in range(1, 7):
        OLD_IMG_DIR = TEMP_DIR / f"{i}" / "images"
        OLD_LABEL_DIR = TEMP_DIR / f"{i}" / "labels"

        for img in sorted(OLD_IMG_DIR.iterdir()):
            label = OLD_LABEL_DIR / img.name
            if not label.exists():
                continue
            new_name = f"tile_{"0" * (6 - len(str(train_val_count)))}{train_val_count}.tif"
            shutil.copy(str(img.resolve()), TRAIN_IMG_DIR / new_name)  # copy image
            shutil.copy(str(label.resolve()), TRAIN_LABEL_DIR / new_name)  # copy label
            train_val_count += 1

        shutil.rmtree(TEMP_DIR / f"{i}")  # delete the temp directory

    # Create test data from images 7, 8
    test_count = 0
    for i in range(7, 9):
        OLD_IMG_DIR = TEMP_DIR / f"{i}" / "images"
        OLD_LABEL_DIR = TEMP_DIR / f"{i}" / "labels"

        for img in sorted(OLD_IMG_DIR.iterdir()):
            label = OLD_LABEL_DIR / img.name
            if not label.exists():
                continue
            new_name = f"tile_{"0" * (6 - len(str(test_count)))}{test_count}.tif"
            shutil.copy(str(img.resolve()), TEST_IMG_DIR / new_name)  # copy image
            shutil.copy(str(label.resolve()), TEST_LABEL_DIR / new_name)  # copy label
            test_count += 1

        shutil.rmtree(TEMP_DIR / f"{i}")  # delete the temp directory

    shutil.rmtree(TEMP_DIR)

    print(f"✅ Tiling completed! Dataset saved at: {GEOAI_DATASET_DIR}")


if __name__ == "__main__":
    main()
