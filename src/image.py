import os
from tqdm import tqdm
import shutil

import cv2
import face_recognition
from inference_sdk import InferenceHTTPClient
from PIL import Image
from retinaface import RetinaFace
from ultralytics import YOLO


# initialize the client
roboflow_client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="HUqmaXJ2N450DbuOPbLY"
)

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


def predict_sibling(image_path):
    result = roboflow_client.infer(image_path, model_id="sibling_detector/1")
    return result


def main():
    # crop faces workflow 
    # input_directory = "/Users/andylee/Projects/scrapbook/data/test_faces"
    # output_directory = "/Users/andylee/Projects/scrapbook/data/test_faces_cropped"
    # crop_faces_from_images(input_directory, output_directory)

    # predict siblings workflow
    input_directory = "/Users/andylee/Projects/scrapbook/data/test_faces_cropped"
    for filename in os.listdir(input_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_directory, filename)
            result = predict_sibling(image_path)
            print(f"Prediction for {filename}: {result}")

            # Extract the predicted classes from the result
            predicted_classes = result.get('predicted_classes', [])

            # Create a new filename with the predicted classes
            predicted_classes_str = "_".join(predicted_classes)
            new_filename = f"{os.path.splitext(filename)[0]}_{predicted_classes_str}.jpg"
            # Ensure the predicted_faces directory exists
            predicted_faces_dir = os.path.join(input_directory, "predicted_faces")
            os.makedirs(predicted_faces_dir, exist_ok=True)

            # Copy and rename the file with the new filename in the predicted_faces directory
            new_image_path = os.path.join(predicted_faces_dir, new_filename)

            # Copy and rename the file with the new filename
            new_image_path = os.path.join(input_directory, new_filename)
            shutil.copy(image_path, new_image_path)

if __name__ == "__main__":
    main()
