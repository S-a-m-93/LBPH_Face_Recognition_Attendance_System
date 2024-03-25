import cv2
import os
import mysql.connector  # Import MySQL connector library
from dotenv import load_dotenv


def generate_dataset():
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            for x, y, w, h in faces:
                cropped_face = img[y : y + h, x : x + w]
                return cropped_face
        else:
            return None

    # Load environment variables from .env file (optional)
    load_dotenv()

    # Access environment variables using os.environ.get()
    mysql_host = os.environ.get("MYSQL_HOST")
    mysql_user = os.environ.get("MYSQL_USER")
    mysql_password = os.environ.get("MYSQL_PASSWORD")
    mysql_database = os.environ.get("MYSQL_DATABASE")

    try:
        mydb = mysql.connector.connect(
            host=mysql_host,
            user=mysql_user,
            password=mysql_password,
            database=mysql_database,
        )
        mycursor = mydb.cursor()
    except mysql.connector.Error as err:
        print("Error connecting to database:", err)
        return

    while True:
        name = input("Enter name: ")
        roll_number = input("Enter roll number: ")

        # Check for existing roll number
        sql = "SELECT * FROM user_data WHERE roll_number=%s"
        val = (roll_number,)
        mycursor.execute(sql, val)
        result = mycursor.fetchone()

        if result:
            print("Roll number already exists.")
            continue

        cap = cv2.VideoCapture(0)
        img_id = 0
        has_inserted_data = False  # Flag to track data insertion

        while img_id < 100:
            ret, frame = cap.read()

            if not ret:
                print("Error reading frame from camera")
                break

            face = face_cropped(frame)
            if face is not None:
                img_id += 1
                face = cv2.resize(face, (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                file_name_path = (
                    "data/user."
                    + str(name)
                    + "."
                    + str(roll_number)
                    + "."
                    + str(img_id)
                    + ".jpg"
                )
                cv2.imwrite(file_name_path, face)
                cv2.putText(
                    face,
                    str(img_id),
                    (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Cropped face", face)

                if not has_inserted_data:
                    try:
                        # Insert data only once per user
                        sql = (
                            "INSERT INTO user_data (name, roll_number) VALUES (%s, %s)"
                        )
                        val = (name, roll_number)
                        mycursor.execute(sql, val)
                        mydb.commit()
                        has_inserted_data = True
                    except mysql.connector.Error as err:
                        print("Error inserting data:", err)

            if cv2.waitKey(1) == 13:
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Finished collecting samples.")

        break  # Exit the loop after successful data collection

    mycursor.close()  # Close database cursor
    mydb.close()  # Close database connection


generate_dataset()
