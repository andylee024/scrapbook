import os
from tqdm import tqdm
from PIL import Image

import cv2
import face_recognition
from retinaface import RetinaFace
from ultralytics import YOLO

def _detect_faces(image):
    """Returns face detections from image"""
    pass

def _classify_siblings(image):
    """Return back list of classified siblings"""
    pass

def _enrich_image_with_metadata(image):
    """Add metadata to image"""
    pass

def _store_enriched_image_to_database(image_data):
    pass


def draw_bounding_boxes(image_path, faces):
    # Read the image
    image = cv2.imread(image_path)

    # Iterate over each detected face
    for face_id, face_data in faces.items():
        # Get the facial area coordinates
        x1, y1, x2, y2 = face_data['facial_area']
        # Draw the rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display the image
    cv2.imshow('Image with Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # save the image with bounding boxes to a new file
    # save the file to scrapbook/data
    cv2.imwrite('/Users/andylee/Projects/scrapbook/data/image_bounding_box.jpg', image)


def crop_faces_from_images(input_dir, output_dir):

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)


    for i, filename in enumerate(tqdm(os.listdir(input_dir), desc="Processing images")):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            image = face_recognition.load_image_file(image_path)

            # Find all face locations in the image
            face_locations = face_recognition.face_locations(image)

            # Iterate over each face found
            for i, face_location in enumerate(face_locations):
                top, right, bottom, left = face_location

                # Crop the face from the image
                face_image = image[top:bottom, left:right]
                pil_image = Image.fromarray(face_image)

                # Save the cropped face to the output directory
                face_filename = f"{os.path.splitext(filename)[0]}_face_{i+1}.jpg"
                pil_image.save(os.path.join(output_dir, face_filename))


def main():
    # Example usage
    input_directory = "/Users/andylee/Desktop/scrapbook_face_in"
    output_directory = "/Users/andylee/Projects/scrapbook/data/output_faces"
    crop_faces_from_images(input_directory, output_directory)

if __name__ == "__main__":
    main()
