import dotenv
import os

from twilio.rest import Client

# Twilio credentials
dotenv.load_dotenv()
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_phone_number = os.getenv("TWILIO_PHONE_NUMBER")

# Initialize Twilio client
client = Client(account_sid, auth_token)

def main():

    # Replace with your phone number
    phone_number_1 = '+447470774593'
    phone_number_2 ='+16262635386'

    client = Client(account_sid, auth_token)
    message = client.messages.create(
        from_=phone_number_2,
        body='Hello boiii',
        to=phone_number_1
    )
    print(message.sid)


if __name__ == "__main__":
    main()
