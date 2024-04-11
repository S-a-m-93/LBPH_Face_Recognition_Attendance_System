import streamlit as st
import cv2
import os
import mysql.connector
import datetime
import numpy as np
from streamlit.secrets import Secrets


# Establish database connection
def connect_to_database():
    """
    Establishes a connection to the database using secrets for credentials.

    Raises:
        ValueError: If required database credentials are missing.

    Returns:
        tuple: A tuple containing the established connection (`mydb`) and cursor (`mycursor`).
    """

    secrets = Secrets()

    mysql_host = secrets["MYSQL_HOST"]
    mysql_user = secrets["MYSQL_USER"]
    mysql_password = secrets["MYSQL_PASSWORD"]
    mysql_database = secrets["MYSQL_DATABASE"]

    if not all([mysql_host, mysql_user, mysql_password, mysql_database]):
        raise ValueError("Missing required database credentials.")

    try:
        mydb = mysql.connector.connect(
            host=mysql_host,
            user=mysql_user,
            password=mysql_password,
            database=mysql_database,
        )
        mycursor = mydb.cursor()
        return mydb, mycursor
    except mysql.connector.Error as err:
        st.error("Error connecting to database.")
        return None, None


# Function to crop the face from an image
def face_cropped(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        for x, y, w, h in faces:
            cropped_face = img[y : y + h, x : x + w]
            return cropped_face
    else:
        return None


# Function to generate dataset
def generate_dataset(name, roll_number, data_dir, mycursor):
    if name.strip() == "" or roll_number.strip() == "":
        st.error("Please enter both name and roll number.")
        return

    cap = cv2.VideoCapture(0)
    img_id = 0

    while img_id < 100:
        ret, frame = cap.read()

        if not ret:
            st.error("Error reading frame from camera")
            break

        face = face_cropped(frame)
        if face is not None:
            img_id += 1
            face = cv2.resize(face, (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            file_name_path = (
                str(data_dir)
                + "/user."
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
            st.image(face, channels="GRAY", use_column_width=True)

    cap.release()
    cv2.destroyAllWindows()
    st.success("Finished collecting samples.")


# Function to train classifier
def train_classifier(data_dir, mycursor):
    """
    Train the classifier using face images stored in `data_dir` directory.

    Args:
        data_dir (str): Path to the directory containing face images.
        mycursor: MySQL cursor object for database operations.
    """
    try:
        mycursor.execute("SELECT id, name, roll_number FROM user_data")
        results = mycursor.fetchall()

        faces = []
        ids = []

        for row in results:
            id = row[0]
            name = row[1]
            roll_number = row[2]

            # Construct the image filename based on the data format
            filename_prefix = f"user.{name}.{roll_number}."
            for image_number in range(1, 101):  # Assuming max 100 images per user
                image_path = os.path.join(
                    data_dir, f"{filename_prefix}{image_number}.jpg"
                )

                if os.path.isfile(image_path):
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        faces.append(img)
                        ids.append(id)

        ids = np.array(ids)

        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.train(faces, ids)
        clf.write("classifier.xml")
        st.success("Classifier trained successfully!")

    except mysql.connector.Error as err:
        st.error("Error accessing database.")


# Function to create date column if not exists
def create_date_column_if_not_exists(date_column, mycursor):
    """
    Create a new date column in the database table if it doesn't exist.

    Args:
        date_column (str): Name of the date column to create.
        mycursor: MySQL cursor object for database operations.
    """
    try:
        sql = f"SHOW COLUMNS FROM user_data LIKE '{date_column}'"
        mycursor.execute(sql)
        result = mycursor.fetchone()
        if not result:
            sql = f"ALTER TABLE user_data ADD COLUMN `{date_column}` VARCHAR(20)"
            mycursor.execute(sql)
            st.success(f"Added new date column: {date_column}")
    except mysql.connector.Error as err:
        st.error("Error accessing database.")


# Function to detect and predict
def detect_and_predict(mycursor, camera_index=0):
    """
    Perform face detection and recognition using a trained classifier.

    Args:
        mycursor: MySQL cursor object for database operations.
        camera_index (int): Index of the camera device to use for video capture.
    """
    try:
        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.read("classifier.xml")
        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        cap = cv2.VideoCapture(camera_index)

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Could not capture frame from camera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for x, y, w, h in faces:
                roi = gray[y : y + h, x : x + w]
                id, pred = clf.predict(roi)
                confidence = int(100 * (1 - pred / 300))

                if confidence > 70:
                    sql = "SELECT name, roll_number FROM user_data WHERE id = %s"
                    val = (id,)
                    mycursor.execute(sql, val)
                    result = mycursor.fetchone()
                    if result:
                        name, roll_number = result
                        st.success(f"Recognized: {name}, Roll Number: {roll_number}")
                else:
                    st.warning("Unknown Person")

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            st.image(frame, channels="BGR", use_column_width=True)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    except mysql.connector.Error as err:
        st.error("Error accessing database.")


# Main function
def main():
    st.title("Face Recognition Attendance System")

    # Connect to database
    mydb, mycursor = connect_to_database()
    if not mydb:
        st.error("Database connection error.")
        return

    # Page navigation
    page = st.sidebar.selectbox(
        "Select Page", ["Train Classifier", "Create Date Column", "Detect and Predict"]
    )

    if page == "Train Classifier":
        st.header("Train Classifier")
        data_dir = st.text_input("Enter path to dataset directory:")
        if st.button("Train"):
            train_classifier(data_dir, mycursor)

    elif page == "Create Date Column":
        st.header("Create Date Column")
        date_column = st.text_input("Enter date column name:")
        if st.button("Create"):
            create_date_column_if_not_exists(date_column, mycursor)

    elif page == "Detect and Predict":
        st.header("Detect and Predict")
        camera_index = st.number_input("Camera Index", value=0, step=1)
        if st.button("Start Detection"):
            detect_and_predict(mycursor, camera_index)

    # Close database connection
    if mydb:
        mycursor.close()
        mydb.close()


if __name__ == "__main__":
    main()
