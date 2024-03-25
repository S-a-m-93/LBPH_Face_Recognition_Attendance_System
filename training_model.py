import os
import cv2
from PIL import Image
import numpy as np
import re
from dotenv import load_dotenv
import mysql.connector


# Function to train the classifier using data from the database
def train_classifier(data_dir):
    try:
        load_dotenv()
        # Use environment variables for database configuration
        mysql_host = os.environ.get("MYSQL_HOST")
        mysql_user = os.environ.get("MYSQL_USER")
        mysql_password = os.environ.get("MYSQL_PASSWORD")
        mysql_database = os.environ.get("MYSQL_DATABASE")

        # Ensure all required environment variables are set
        if not all([mysql_host, mysql_user, mysql_password, mysql_database]):
            raise ValueError(
                "Missing required environment variables for database connection."
            )

        mydb = mysql.connector.connect(
            host=mysql_host,
            user=mysql_user,
            password=mysql_password,
            database=mysql_database,
        )  # Connect to the database
        mycursor = mydb.cursor()

        sql = "SELECT id, name, roll_number FROM user_data"  # Retrieve user data
        mycursor.execute(sql)
        results = mycursor.fetchall()

        faces = []
        ids = []

        for row in results:
            try:
                id = row[0]
                name = row[1]
                roll_number = row[2]

                # Construct the image filename based on the data format
                filename = f"user.{name}.{roll_number}."  # Common prefix

                # Loop through potential image numbers (adjust if needed)
                for image_number in range(1, 101):  # Assuming max 100 images per user
                    image_path = os.path.join(
                        data_dir, filename + f"{image_number}.jpg"
                    )

                    # Check if the image file exists
                    if not os.path.isfile(image_path):
                        continue  # Skip if image not found

                    # Read the image from the file system
                    img = cv2.imread(
                        image_path, cv2.IMREAD_GRAYSCALE
                    )  # Read as grayscale

                    if img is None:  # Check if image is valid
                        print(f"Error: Could not read image {image_path}")
                        continue

                    faces.append(img)
                    ids.append(id)

            except Exception as e:
                print(f"Error processing user {name}: {e}")
                continue

        ids = np.array(ids)

        try:
            clf = cv2.face.LBPHFaceRecognizer_create()
            clf.train(faces, ids)
            clf.write("classifier.xml")
            print("Classifier trained successfully!")
        except cv2.error as e:
            print(f"OpenCV error during training: {e}")

    except mysql.connector.Error as err:
        print("Error connecting to database:", err)

    finally:
        if mydb:
            mycursor.close()
            mydb.close()


# Provide path to the image data directory (replace with your actual path)
data_dir = "data"

train_classifier(data_dir)
