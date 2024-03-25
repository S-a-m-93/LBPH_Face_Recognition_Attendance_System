import cv2
import mysql.connector
import os
from dotenv import load_dotenv
import datetime

load_dotenv()

# Access environment variables using os.environ.get()
mysql_host = os.environ.get("MYSQL_HOST")
mysql_user = os.environ.get("MYSQL_USER")
mysql_password = os.environ.get("MYSQL_PASSWORD")
mysql_database = os.environ.get("MYSQL_DATABASE")

# Connect to the database
mydb = mysql.connector.connect(
    host=mysql_host,
    user=mysql_user,
    password=mysql_password,
    database=mysql_database,
)


def create_date_column_if_not_exists(date_column):
    """
    Check if the given date_column exists in the 'user_data' table.
    If not, create the column.
    """
    try:
        mycursor = mydb.cursor()
        sql = f"SHOW COLUMNS FROM user_data LIKE '{date_column}'"
        mycursor.execute(sql)
        result = mycursor.fetchone()
        if not result:  # date_column does not exist
            sql = f"ALTER TABLE user_data ADD COLUMN `{date_column}` VARCHAR(20)"
            mycursor.execute(sql)
            mydb.commit()
    except mysql.connector.Error as error:
        print("Database error:", error)


def detect_and_predict(camera_index=0):
    """
    This function opens the webcam, detects faces, predicts user ID,
    retrieves name and roll number from the database, and marks attendance with timestamp.
    """
    cap = cv2.VideoCapture(camera_index)
    marked_users = set()  # Set to store IDs of users marked present today

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not capture frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for x, y, w, h in faces:
            roi = gray[y : y + h, x : x + w]
            id, pred = classifier.predict(roi)
            confidence = int(100 * (1 - pred / 300))

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            if confidence > 70:
                try:
                    mycursor = mydb.cursor()

                    # Get name and roll number
                    sql = "SELECT name, roll_number FROM user_data WHERE id = %s"
                    val = (id,)
                    mycursor.execute(sql, val)
                    result = mycursor.fetchone()
                    name, roll_number = result

                    # Get current date and timestamp
                    todays_date = datetime.date.today().strftime("%Y-%m-%d")
                    current_time = datetime.datetime.now().strftime("%H:%M:%S")

                    # Create or check date column for today
                    create_date_column_if_not_exists(todays_date)

                    # Check if the user is already marked for today
                    if id in marked_users:
                        message = "Marked"
                    else:
                        # Update attendance with timestamp if not already marked
                        sql = f"UPDATE user_data SET `{todays_date}` = '{current_time}' WHERE id = %s AND `{todays_date}` IS NULL"
                        val = (id,)
                        mycursor.execute(sql, val)
                        mydb.commit()

                        marked_users.add(id)  # Add user ID to marked list
                        message = "Marked"

                    # Display message
                    cv2.putText(
                        frame,
                        f"Name: {name}, Roll: {roll_number} ({message})",
                        (x + 2, y + h + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )

                except mysql.connector.Error as error:
                    print("Database error:", error)

            else:
                cv2.putText(
                    frame,
                    "Unknown",
                    (x + 2, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# Load classifier and cascade models
classifier = cv2.face.LBPHFaceRecognizer_create()
classifier.read("classifier.xml")
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Start face detection
detect_and_predict()

print("Prediction finished.")
