# LBPH Face Recognition Attendance System

This project is a face recognition attendance system that uses OpenCV with LBPH (Local Binary Patterns Histograms) for face recognition, along with a MySQL database to store attendance data. The user interface (UI) is built using Streamlit, a framework for creating web applications with Python.

## Features

- Detects faces using LBPH algorithm.
- Stores user data (name, roll number) in a MySQL database.
- Captures and stores images of users for training the recognition model.
- Marks attendance with timestamps in the database.

## Requirements

To run this project locally, you will need to have the following installed:

- Python 3.x
- MySQL Server (for database storage)

## Local Environment Setup

Follow these steps to set up your local development environment:

### 1. Clone the Repository

First, clone this GitHub repository to your local machine:

```bash
git clone https://github.com/S-a-m-93/LBPH_Face_Recognition_Attendance_System.git
cd LBPH_Face_Recognition_Attendance_System
```

### 2. Set Up a Virtual Environment

It's recommended to use a virtual environment to manage project dependencies. Create and activate a virtual environment using venv (for Python 3) or virtualenv:

Using venv (Python 3):

```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate      # On Windows
```

Using virtualenv:

```bash
pip install virtualenv
virtualenv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate      # On Windows
```

### 3. Install Python Dependencies

With the virtual environment activated, install the required Python packages using pip and the requirements.txt file:

```bash
pip install -r requirements.txt
```

### 4. Set Up MySQL Database

Install and set up a MySQL database to store attendance data. You can use MySQL Workbench or any other MySQL client to create the necessary database and tables. Here is an example of SQL commands to set up the database schema:

```bash
CREATE DATABASE face_recognition;
USE face_recognition;
CREATE TABLE user_data(
  id INT AUTO_INCREMENT PRIMARY KEY,  /* Auto-incrementing ID as primary key */
  name VARCHAR(255) NOT NULL,
  roll_number INT NOT NULL,
  UNIQUE(roll_number)  /* Enforce unique roll number constraint */
);
```

### 5. Configure Database Connection

Create a .streamlit folder in the project root directory, and inside that folder, create a secrets.toml file with your MySQL database configuration:

```bash
mkdir .streamlit
echo "[database]" >> .streamlit/secrets.toml
echo "MYSQL_HOST = 'localhost'" >> .streamlit/secrets.toml
echo "MYSQL_USER = 'your_mysql_username'" >> .streamlit/secrets.toml
echo "MYSQL_PASSWORD = 'your_mysql_password'" >> .streamlit/secrets.toml
echo "MYSQL_DATABASE = 'face_recognition'" >> .streamlit/secrets.toml
```

Replace 'your_mysql_username' and 'your_mysql_password' with your actual MySQL database username and password.

### 6. Run the Application

After setting up the environment and database, you can run the Streamlit app:

```bash
streamlit run app.py
```

This command will start the Streamlit server locally. Open a web browser and go to the URL displayed in the terminal to use the face recognition attendance system.

## Usage

- When you run the Streamlit app, you should see the interface for taking attendance.
- The app will use your webcam to capture faces and recognize them using LBPH face recognition.
- Detected faces will be compared against the database of students.
- The system will mark attendance for recognized students in the MySQL database.

By following these steps, you should be able to set up and run the face recognition attendance system locally on your machine within a virtual environment. This helps ensure a clean and isolated environment for your project's dependencies. If you have any questions or encounter issues during setup, feel free to reach out for assistance!
