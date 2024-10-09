from ultralytics import YOLO
from retinaface import RetinaFace
import cv2

def main():
    # YOLO model
    # weights_path = "/Users/andylee/Projects/scrapbook/data/yolov10n.pt"
    # model = YOLO(weights_path)  # load a pretrained model (recommended for training)
    # results = model(image_path)  # list of Results objects
    img1 = "/Users/andylee/Downloads/dtb_team_dinner.jpeg"
    image_path = "/Users/andylee/Desktop/andy_fphotos/IMG_2051.JPG"

    response = RetinaFace.detect_faces(img1)
    print(response)

    # plot the coordinates of detected faces 
    draw_bounding_boxes(img1, response)


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


if __name__ == "__main__":
    main()
