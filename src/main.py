import dotenv
import os
import random
from PIL import Image

from twilio.rest import Client


client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))

def _pick_random_image_path(image_directory):
    # Filter for files with .jpg, .jpeg, .png extensions
    acceptable_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(tuple(acceptable_extensions))]
    return os.path.join(image_directory, random.choice(image_files))


def main():

    # Replace with your phone number
    phone_number_1 = '+447470774593'
    phone_number_2 ='+16262635386'

    image_directory = "/Users/andylee/Desktop/andy_fphotos"
    image_path = _pick_random_image_path(image_directory)
    print(image_path)

    # display the image
    # image = Image.open(image_path)
    # image.show()

    image_link = "https://drive.google.com/file/d/18ljnSX2aVG0LRcYONagXl8FFx7nrVEGo/view?usp=sharing"

    message = client.messages.create(
        from_=phone_number_2,
        body='Hello boiii w/ image',
        to=phone_number_1,
        media_url=[image_link]
    )
    print(message.sid)
if __name__ == "__main__":
    main()
