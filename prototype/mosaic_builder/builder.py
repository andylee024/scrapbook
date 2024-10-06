import os
import sys
import glob
import random
from heapq import heappush, heappushpop

from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
from sklearn.cluster import KMeans

import numpy as np
from PIL import Image
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from components.mosaic_builder.image_segmentor import ImageSegmenter


class MosaicBuilder:
    """
    A class to build mosaic images from a reference image and a set of candidate tiles.
    """

    def __init__(
            self,
            reference_image_path: str,
            candidate_tiles_dir: str,
            mosaic_config: Dict[str, Any]
    ):
        """
        Initialize the MosaicBuilder with the given parameters.

        :param reference_image_path: Path to the reference image
        :param candidate_tiles_dir: Directory containing candidate tile images
        :param mosaic_config: Configuration dictionary for mosaic building
        """
        self.base_image = self.load_image(reference_image_path)
        self.candidate_tiles_dir = candidate_tiles_dir
        self.mosaic_config = mosaic_config
        self.min_tile_size = mosaic_config.get("min_tile_size", 10)
        self.max_tile_size = mosaic_config.get("max_tile_size", 100)
        self.max_num_tiles = mosaic_config.get("max_num_tiles", 1000)
        self.tile_images = self.load_tile_images()

    def load_image(self, image_path: str) -> np.ndarray:
        """Load an image from the given path."""
        # Corrected the method calls for cv2
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # make the image 5 times larger
        return cv2.resize(image_rgb, None, fx=10, fy=10)  # Use None for the size parameter

    def load_tile_images(self) -> List[Tuple[Image.Image, Tuple[int, int]]]:
        """Load and preprocess tile images from the candidate tiles directory."""
        
        tile_image_paths = []
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            tile_image_paths.extend(glob.glob(os.path.join(self.candidate_tiles_dir, ext.lower())))
        
        tile_images = []
    

        for tile_file in tile_image_paths:
            tile_path = os.path.join(self.candidate_tiles_dir, tile_file)
            tile_image = Image.open(tile_path)
            original_size = tile_image.size
            # Resize the image to fit within our size constraints while maintaining aspect ratio
            tile_image.thumbnail((self.max_tile_size, self.max_tile_size))
            if min(tile_image.size) < self.min_tile_size:
                continue
            tile_images.append((tile_image, original_size))
        return tile_images

    def calculate_colour_metrics(self, image: Image.Image) -> np.ndarray:
        """Calculate the average color of the given image."""
        img = np.array(image)
        avg_color = np.mean(img, axis=(0, 1))[:3]
        return avg_color

    def calculate_tile_metrics(self, image: Image.Image) -> Dict[str, Any]:
        """Calculate metrics for a given tile image."""
        avg_color = self.calculate_colour_metrics(image)
        return {"color": avg_color, "size": image.size}


    def find_best_tile(
        self, target_color: np.ndarray, target_size: Tuple[int, int]
    ) -> List[Tuple[Image.Image, Tuple[int, int]]]:
        """Find the top 5 best matching tiles for the given target color and size."""
        best_tiles = []
        for tile, original_size in self.tile_images:
            color_diff = np.linalg.norm(self.calculate_colour_metrics(tile) - target_color)
            size_diff = abs(tile.size[0] * tile.size[1] - target_size[0] * target_size[1])
            # Adjust this factor to balance color vs size importance
            score = color_diff + size_diff * 0.01
        
            if len(best_tiles) < 5:
                heappush(best_tiles, (-score, (tile, original_size)))
            else:
                heappushpop(best_tiles, (-score, (tile, original_size)))
        # Randomly select a tile from the top 5
        return random.choice([tile_info for _, tile_info in sorted(best_tiles, reverse=True)])
    
    def augment_tile(self, tile: Image.Image, block: np.ndarray) -> Image.Image:
        """Augment the tile's color to be closer to the block while preserving its original characteristics."""
        # Convert the tile to a numpy array
        tile_array = np.array(tile).astype(np.float64)
        
        # Calculate the average color of the block
        block_avg_color = np.mean(block, axis=(0, 1))

        # Calculate the average color of the tile
        tile_avg_color = np.mean(tile_array, axis=(0, 1))

        # Calculate the color difference
        color_diff = block_avg_color - tile_avg_color

        # Calculate the adjustment factor (you can adjust this value to control the strength of the effect)
        adjustment_factor = 1

        # Apply the color adjustment using alpha blending
        blended_color = tile_array * (1 - adjustment_factor) + (tile_array + color_diff) * adjustment_factor

        # Ensure the resulting values are within the valid range
        augmented_tile = np.clip(blended_color, 0, 255).astype(np.uint8)
        
        # Convert the augmented tile back to an image (preserving the original mode)
        augmented_tile = Image.fromarray(augmented_tile, mode=tile.mode)
        
        return augmented_tile
        

    def segment_image(self, image: np.ndarray) -> list:
        """Segment the image into blocks of consistent color."""
        segmenter = ImageSegmenter(image)
        rectangles = segmenter.build_rectangles()
        
        return rectangles

    def build(self, output_path: str) -> None:
        """Build the mosaic image and save it to the specified output path."""
        base_height, base_width = self.base_image.shape[:2]
        mosaic_image = Image.new('RGB', (base_width, base_height))

        # Segment the image into blocks of consistent color
        blocks = self.segment_image(self.base_image)


        for block in blocks:
            # Unpack the block
            x, y, block_width, block_height = block

            # Get the average color of the current block
            block = self.base_image[y:y+block_height, x:x+block_width]
            block_color = np.mean(block, axis=(0, 1))

            # Find the best matching tile
            best_tile, original_size = self.find_best_tile(
                block_color, (block_width, block_height)
            )

            # Augment the best tile to be closer to block
            best_tile = self.augment_tile(best_tile, block)

            # Maintain aspect ratio while resizing
            best_tile_aspect_ratio = original_size[0] / original_size[1]
            block_aspect_ratio = block_width / block_height

            if best_tile_aspect_ratio > block_aspect_ratio:
                # Crop width
                new_width = int(block_height * best_tile_aspect_ratio)
                best_tile = best_tile.resize((new_width, block_height), Image.LANCZOS)
                left = (new_width - block_width) / 2
                best_tile = best_tile.crop((left, 0, left + block_width, block_height))
            else:
                # Crop height
                new_height = int(block_width / best_tile_aspect_ratio)
                best_tile = best_tile.resize((block_width, new_height), Image.LANCZOS)
                top = (new_height - block_height) / 2
                best_tile = best_tile.crop((0, top, block_width, top + block_height))

            # Paste the tile onto the mosaic
            mosaic_image.paste(best_tile, (x, y))
            
        mosaic_image.save(output_path)

if __name__ == "__main__":
    mosaic_builder = MosaicBuilder(
        reference_image_path='/Users/hamed/Desktop/iCloud Photos from Andy Lee/IMG_1868.JPG',
        candidate_tiles_dir='/Users/hamed/Desktop/iCloud Photos from Andy Lee/',
        mosaic_config={
            "min_tile_size": 10,
            "max_tile_size": 30,
            "max_num_tiles": 1000
        }
    )
    
    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path.cwd() / f"runs/{current_timestamp}.jpg"

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    mosaic_builder.build(output_path=output_path)