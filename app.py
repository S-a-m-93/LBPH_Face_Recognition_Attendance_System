import streamlit as st
import cv2
import os
import mysql.connector
from dotenv import load_dotenv
import datetime
import numpy as np


# Connect to the database
load_dotenv()
mysql_host = os.environ.get("MYSQL_HOST")
mysql_user = os.environ.get("MYSQL_USER")
mysql_password = os.environ.get("MYSQL_PASSWORD")
mysql_database = os.environ.get("MYSQL_DATABASE")
if not all([mysql_host, mysql_user, mysql_password, mysql_database]):
    raise ValueError("Missing required environment variables for database connection.")

try:
    mydb = mysql.connector.connect(
        host=mysql_host,
        user=mysql_user,
        password=mysql_password,
        database=mysql_database,
    )
    mycursor = mydb.cursor()
except mysql.connector.Error as err:
    st.error("Error connecting to database.")


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
def generate_dataset(name, roll_number, data_dir):
    if name.strip() == "" or roll_number.strip() == "":
        st.error("Please enter both name and roll number.")
        return

    while True:
        sql = "SELECT * FROM user_data WHERE roll_number=%s"
        val = (roll_number,)
        mycursor.execute(sql, val)
        result = mycursor.fetchone()

        if result:
            st.error("Roll number already exists.")
            break

        cap = cv2.VideoCapture(0)
        img_id = 0
        has_inserted_data = False

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
                if not has_inserted_data:
                    try:
                        sql = (
                            "INSERT INTO user_data (name, roll_number) VALUES (%s, %s)"
                        )
                        val = (name, roll_number)
                        mycursor.execute(sql, val)
                        mydb.commit()
                        has_inserted_data = True
                    except mysql.connector.Error as err:
                        st.error("Error inserting data.")

            if cv2.waitKey(1) == 13:
                break

        cap.release()
        cv2.destroyAllWindows()
        st.success("Finished collecting samples.")

        break
    mycursor.close()
    mydb.close()

    pass


# Function to train classifier
def train_classifier(data_dir):
    try:
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
                        st.error(f"Error: Could not read image {image_path}")
                        continue

                    faces.append(img)
                    ids.append(id)

            except Exception as e:
                st.error(f"Error processing user {name}: {e}")
                continue

        ids = np.array(ids)

        try:
            clf = cv2.face.LBPHFaceRecognizer_create()
            clf.train(faces, ids)
            clf.write("classifier.xml")
            st.success("Classifier trained successfully!")
        except cv2.error as e:
            st.error(f"OpenCV error during training: {e}")

    except mysql.connector.Error as err:
        st.error("Error connecting to database.")

    finally:
        if mydb:
            mycursor.close()
            mydb.close()

    pass


# Function to create date column if not exists
def create_date_column_if_not_exists(date_column):
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
        st.error("Database error:", error)
    pass


# Function to detect and predict
def detect_and_predict(camera_index=0):
    classifier = cv2.face.LBPHFaceRecognizer_create()
    classifier.read("classifier.xml")
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Start face detection
    cap = cv2.VideoCapture(camera_index)
    marked_users = set()  # Set to store IDs of users marked present today

    st.header("Face Recognition")

    ret, frame = cap.read()
    if not ret:
        st.error("Error: Could not capture frame from camera.")
        cap.release()
        cv2.destroyAllWindows()
        return

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
                current_datetime = datetime.datetime.now()
                current_time = current_datetime.strftime("%H:%M:%S")
                todays_date = current_datetime.strftime("%Y-%m-%d")

                # Create or check date column for today
                create_date_column_if_not_exists(todays_date)

                # Check if the user is already marked for today
                sql = f"SELECT `{todays_date}` FROM user_data WHERE id = %s"
                val = (id,)
                mycursor.execute(sql, val)
                result = mycursor.fetchone()
                first_attendance_time = result[0] if result else None

                # Update attendance only if not already marked today
                if first_attendance_time is None:
                    sql = f"UPDATE user_data SET `{todays_date}` = '{current_time}' WHERE id = %s"
                    val = (id,)
                    mycursor.execute(sql, val)
                    mydb.commit()

                    marked_users.add(id)  # Add user ID to marked list

                # Display attendance details using first_attendance_time or current_time
                st.subheader("Attendance Details")
                display_time = first_attendance_time or current_time
                message = "Marked"
                attendance_details = f"""**Name:** {name}  
                **Roll:** {roll_number}  
                **Attendance:** {message}  
                **Time:** {display_time}  
                **Date:** {todays_date}"""

                st.markdown(attendance_details)
                st.success("Attendance marked successfully!")

            except mysql.connector.Error as error:
                st.error("Database error.")
        else:
            # Unrecognized face
            st.error(
                "**You are not recognized.** Please go to the generate dataset page and then train the model."
            )
    st.image(frame, channels="BGR", use_column_width=True)

    cap.release()
    cv2.destroyAllWindows()


# Main function
def main():
    st.title("Face Recognition Attendance System")

    # Navigation
    page = st.sidebar.selectbox(
        "Select Page", ["Generate Dataset", "Train Classifier", "Start Attendance"]
    )

    if page == "Generate Dataset":
        st.header("Generate Dataset")
        st.write("This page is used to generate dataset.")
        data_dir = st.text_input("Enter path for where you want to store your data:")
        name = st.text_input("Enter name:")
        roll_number = st.text_input("Enter roll number:")

        if st.button("Generate"):
            st.write(
                "Please look at the camera and ensure that you are in a well lit place."
            )
            generate_dataset(name, roll_number, data_dir)
            st.success("Your pictures have been collected.")
            st.write("Please move on to the training classifier page.")

    elif page == "Train Classifier":
        st.header("Train Classifier")
        st.write("This page is used to train the classifier.")
        data_dir = st.text_input("Enter path to dataset directory")
        if st.button("Train"):
            train_classifier(data_dir)
            st.success("Classifier trained successfully!")

    elif page == "Start Attendance":
        st.header("Start Attendance")
        st.write("This page is used to start attendance.")
        st.write("Please ensure there is only one person in the frame.")
        camera_index = st.number_input("Camera Index", value=0, step=1)
        if st.button("Start"):
            detect_and_predict(camera_index)


if __name__ == "__main__":
    main()
