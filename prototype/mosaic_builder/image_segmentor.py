import numpy as np
import cv2
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw

class ImageSegmenter:
    def __init__(
            self, 
            base_image: np.ndarray,
            num_colors: int = 10, 
            color_variance_threshold: float = 250.0,
            min_block_size: int = 100,
            max_block_size: int = 250  # New parameter
        ):
        """Initialize with the base image and number of dominant colors."""
        self.base_image = base_image
        self.num_colors = num_colors
        self.color_variance_threshold = color_variance_threshold
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size  # New attribute
        self.rectangles = []

    def color_variance(self, region: np.ndarray) -> float:
        """Calculate the color variance within a region."""
        return np.var(region.reshape(-1, 3), axis=0).mean()

    def segment_image(self, image: np.ndarray, x: int, y: int, width: int, height: int) -> None:
        """Recursively segment the image into consistent color blocks (rectangles)."""
        # Check if the current block is smaller than the minimum size
        if width <= self.min_block_size or height <= self.min_block_size:
            self.rectangles.append((x, y, width, height))
            return

        # Check if the current block is larger than the maximum size
        if width > self.max_block_size or height > self.max_block_size:
            # Split the block regardless of color variance
            half_width = width // 2
            half_height = height // 2
            
            # Recursively divide into 4 subregions
            self.segment_image(image, x, y, half_width, half_height)
            self.segment_image(image, x + half_width, y, width - half_width, half_height)
            self.segment_image(image, x, y + half_height, half_width, height - half_height)
            self.segment_image(image, x + half_width, y + half_height, width - half_width, height - half_height)
            return

        region = image[y:y + height, x:x + width]
        
        # Calculate the color variance within the region
        variance = self.color_variance(region)
        
        if variance < self.color_variance_threshold:
            # If the region is color-consistent, save the rectangle
            self.rectangles.append((x, y, width, height))
        else:
            # Otherwise, split the region into smaller rectangles and recurse
            half_width = width // 2
            half_height = height // 2
            
            # Recursively divide into 4 subregions
            # Top-left
            self.segment_image(
                image, x, y, half_width, half_height
            )  
            # Top-right
            self.segment_image(
                image, x + half_width, y, width - half_width, half_height
            ) 
            # Bottom-left
            self.segment_image(
                image, x, y + half_height, half_width, height - half_height
            )
            # Bottom-right
            self.segment_image(
                image, x + half_width, y + half_height, width - half_width, height - half_height
            )

    def build_rectangles(self):
        """Segment the entire base image into rectangles."""
        height, width = self.base_image.shape[:2]
        self.segment_image(self.base_image, 0, 0, width, height)

        return self.rectangles

    def draw_rectangles(self, output_path: str) -> None:
        """Draw the rectangles on a blank canvas and save the output."""
        height, width = self.base_image.shape[:2]
        result_image = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(result_image)

        # Draw all rectangles
        for rect in self.rectangles:
            x, y, w, h = rect
            region_color = np.mean(self.base_image[y:y+h, x:x+w].reshape(-1, 3), axis=0)
            color = tuple([int(c) for c in region_color])
            draw.rectangle([x, y, x+w, y+h], fill=color)
        
        # Draw a box around each rectangle
        for rect in self.rectangles:
            x, y, w, h = rect
            draw.rectangle([x, y, x+w, y+h], outline=(0, 0, 0), width=2)    

        # Save the result image
        result_image.save(output_path)

# Example usage:
# Load an image as a NumPy array (e.g., using OpenCV)
base_image = cv2.imread('/Users/hamed/Desktop/iCloud Photos from Andy Lee/IMG_1868.JPG')
base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Initialize the segmenter and process the image
segmenter = ImageSegmenter(base_image, num_colors=10, color_variance_threshold=250.0, min_block_size=20, max_block_size=100)
rectangles = segmenter.build_rectangles()

# Draw and save the resulting image with segmented rectangles
segmenter.draw_rectangles('output_mosaic.png')
