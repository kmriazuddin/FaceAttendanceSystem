# 1. INSTALL LIBRARIES
!pip install face_recognition -q
!pip install opencv-python-headless -q

print("✅ Libraries installed successfully!")

# 2. IMPORT LIBRARIES
import face_recognition
import cv2
import numpy as np
import csv
import base64
import pandas as pd

from datetime import datetime

from IPython.display import display, Javascript
from google.colab.output import eval_js
from google.colab import drive, files


print("✅ Libraries imported successfully!")

# 3. GOOGLE DRIVE SETUP
drive.mount('/content/drive')


BASE_DIR = '/content/drive/MyDrive/FaceAttendanceSystem'

KNOWN_FACES_DIR = os.path.join(
    BASE_DIR,
    'known_faces'
)

ATTENDANCE_FILE = os.path.join(
    BASE_DIR,
    'attendance.csv'
)


# Create folders
os.makedirs(
    KNOWN_FACES_DIR,
    exist_ok=True
)


# Create attendance CSV
if not os.path.exists(ATTENDANCE_FILE):

    with open(
        ATTENDANCE_FILE,
        'w',
        newline=''
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            'Name',
            'Date',
            'Time'
        ])


print()
print("============================================")
print("GOOGLE DRIVE SETUP")
print("============================================")
print()
print("Known faces:")
print(KNOWN_FACES_DIR)
print()
print("Attendance:")
print(ATTENDANCE_FILE)

# 4. REGISTER / UPLOAD STUDENT IMAGES
# Example:
# Riaz.jpg
# Filename becomes student name.
# One image = one student.
# Image should contain only ONE face.
print()
print("============================================")
print("REGISTER STUDENTS")
print("============================================")
print()
print("Upload student reference images.")
print()
print("Example:")
print("Riaz.jpg")
print()

uploaded = files.upload()


for filename, content in uploaded.items():

    if not filename.lower().endswith(
        ('.jpg', '.jpeg', '.png')
    ):

        print(
            f"❌ Skipped: {filename}"
        )

        continue


    destination = os.path.join(
        KNOWN_FACES_DIR,
        filename
    )


    with open(
        destination,
        'wb'
    ) as f:

        f.write(content)


    print(
        f"✅ Saved: {filename}"
    )

# 5. SHOW REGISTERED STUDENTS
print()
print("============================================")
print("REGISTERED STUDENTS")
print("============================================")

registered_files = []

for filename in sorted(
    os.listdir(KNOWN_FACES_DIR)
):

    if filename.lower().endswith(
        ('.jpg', '.jpeg', '.png')
    ):

        registered_files.append(
            filename
        )


if len(registered_files) == 0:

    print(
        "❌ No students registered."
    )

else:

    for index, filename in enumerate(
        registered_files,
        start=1
    ):

        name = os.path.splitext(
            filename
        )[0]

        print(
            f"{index}. {name}"
        )


print()
print(
    "Total registered images:",
    len(registered_files)
)
