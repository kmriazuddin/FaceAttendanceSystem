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

# 10. PROCESS VIDEO
#     FACE RECOGNITION + CONSISTENCY + LIVENESS

def process_verification_video(
    video_path,
    known_encodings,
    known_names
):


    cap = cv2.VideoCapture(
        video_path
    )


    if not cap.isOpened():

        print(
            "❌ Cannot open video."
        )

        return None


    recognized_names = []

    face_distances = []

    x_positions = []

    frame_count = 0


    print()
    print("============================================")
    print("PROCESSING VERIFICATION")
    print("============================================")
    print()


    while True:

        ret, frame = cap.read()


        if not ret:

            break


        frame_count += 1


        # Process every 3rd frame
        if frame_count % 3 != 0:

            continue


        name, distance, location = (
            verify_face_frame(
                frame,
                known_encodings,
                known_names
            )
        )


        # ----------------------------------------
        # MULTIPLE FACES
        # ----------------------------------------

        if name == "MULTIPLE":

            print(
                "❌ Multiple faces detected."
            )

            cap.release()

            return None


        # ----------------------------------------
        # UNKNOWN FACE
        # ----------------------------------------

        if name == "UNKNOWN":

            continue


        # ----------------------------------------
        # NO FACE
        # ----------------------------------------

        if name is None:

            continue


        # ----------------------------------------
        # RECOGNIZED USER
        # ----------------------------------------

        recognized_names.append(
            name
        )


        if distance is not None:

            face_distances.append(
                distance
            )


        # ----------------------------------------
        # HEAD POSITION
        # ----------------------------------------

        if location is not None:

            top, right, bottom, left = (
                location
            )


            center_x = (
                left + right
            ) / 2


            x_positions.append(
                center_x
            )


    cap.release()


    # ========================================================
    # CHECK 1: FACE FOUND
    # ========================================================

    if len(recognized_names) == 0:

        print(
            "❌ No recognized face."
        )

        return None


    # ========================================================
    # CHECK 2: MOST COMMON USER
    # ========================================================

    unique_names, counts = np.unique(
        recognized_names,
        return_counts=True
    )


    best_index = np.argmax(
        counts
    )


    best_name = unique_names[
        best_index
    ]


    recognition_count = counts[
        best_index
    ]


    print(
        "Recognized user:",
        best_name
    )


    print(
        "Recognition count:",
        recognition_count
    )


    # ========================================================
    # CHECK 3: RECOGNITION CONSISTENCY
    # ========================================================

    if (
        recognition_count
        <
        MIN_RECOGNITION_COUNT
    ):

        print(
            "❌ Recognition consistency failed."
        )

        return None


    # ========================================================
    # CHECK 4: FACE DISTANCE
    # ========================================================

    if len(face_distances) > 0:

        average_distance = np.mean(
            face_distances
        )


        best_distance = np.min(
            face_distances
        )


        print(
            f"Average face distance: "
            f"{average_distance:.4f}"
        )


        print(
            f"Best face distance: "
            f"{best_distance:.4f}"
        )


        if (
            average_distance
            >
            FACE_TOLERANCE
        ):

            print(
                "❌ Face distance check failed."
            )

            return None


    # ========================================================
    # CHECK 5: HEAD MOVEMENT / LIVENESS
    # ========================================================

    if len(x_positions) < 5:

        print(
            "❌ Not enough movement data."
        )

        return None


    horizontal_movement = (
        max(x_positions)
        -
        min(x_positions)
    )


    print(
        f"Horizontal movement: "
        f"{horizontal_movement:.2f}"
    )


    if (
        horizontal_movement
        <
        MIN_HEAD_MOVEMENT
    ):

        print()
        print(
            "============================================"
        )
        print(
            "❌ LIVENESS FAILED"
        )
        print(
            "============================================"
        )
        print()
        print(
            "Please move your head LEFT and RIGHT."
        )

        return None


    # ========================================================
    # ALL CHECKS PASSED
    # ========================================================

    print()
    print(
        "============================================"
    )
    print(
        "✅ FACE VERIFICATION PASSED"
    )
    print(
        "============================================"
    )


    print(
        "User:",
        best_name
    )


    print(
        "Recognition:",
        "PASSED"
    )


    print(
        "Liveness:",
        "PASSED"
    )


    return best_name

# 11. ATTENDANCE FUNCTIONS
# ============================================================

def already_marked_today(name):


    today = datetime.now().strftime(
        '%Y-%m-%d'
    )


    if not os.path.exists(
        ATTENDANCE_FILE
    ):

        return False


    with open(
        ATTENDANCE_FILE,
        'r',
        newline=''
    ) as f:


        reader = csv.reader(f)


        # Skip header
        next(
            reader,
            None
        )


        for row in reader:

            if (
                len(row) >= 2
                and
                row[0] == name
                and
                row[1] == today
            ):

                return True


    return False



def mark_attendance(name):


    # Already marked
    if already_marked_today(
        name
    ):

        print()
        print(
            "============================================"
        )

        print(
            f"⚠️ {name} already marked today."
        )

        print(
            "============================================"
        )

        return


    now = datetime.now()


    with open(
        ATTENDANCE_FILE,
        'a',
        newline=''
    ) as f:


        writer = csv.writer(
            f
        )


        writer.writerow([
            name,
            now.strftime(
                '%Y-%m-%d'
            ),
            now.strftime(
                '%H:%M:%S'
            )
        ])


    print()
    print(
        "============================================"
    )
    print(
        "✅ ATTENDANCE MARKED"
    )
    print(
        "============================================"
    )


    print(
        "Student:",
        name
    )


    print(
        "Date:",
        now.strftime(
            '%Y-%m-%d'
        )
    )


    print(
        "Time:",
        now.strftime(
            '%H:%M:%S'
        )
    )


# ============================================================
# 12. START VERIFICATION
# ============================================================

if len(known_names) == 0:

    print()
    print(
        "❌ No valid students found."
    )

    print(
        "Upload student images first."
    )

else:

    print()
    print(
        "============================================"
    )
    print(
        "START FACE VERIFICATION"
    )
    print(
        "============================================"
    )


    print()
    print(
        "Instructions:"
    )


    print(
        "1. Only ONE person should be visible."
    )


    print(
        "2. Look directly at the camera."
    )


    print(
        "3. Slowly move your head LEFT."
    )


    print(
        "4. Slowly move your head RIGHT."
    )


    print(
        "5. Keep your face visible."
    )


    print()


    # Capture webcam video
    video_path = capture_webcam_video(
        filename='verification.webm',
        duration=8
    )


    print()
    print(
        "✅ Webcam video captured."
    )


# ============================================================
# 13. RUN VERIFICATION
# ============================================================

if len(known_names) > 0:

    verified_user = (
        process_verification_video(
            video_path,
            known_encodings,
            known_names
        )
    )


    print()


    if verified_user is not None:

        print(
            "============================================"
        )

        print(
            "🎉 VERIFIED"
        )

        print(
            "Student:",
            verified_user
        )

        print(
            "============================================"


        )

    else:

        print(
            "============================================"
        )

        print(
            "❌ VERIFICATION FAILED"
        )

        print(
            "Attendance will NOT be marked."
        )

        print(
            "============================================"


        )


# ============================================================
# 14. MARK ATTENDANCE
# ============================================================

if (
    len(known_names) > 0
    and
    verified_user is not None
):

    mark_attendance(
        verified_user
    )

else:

    print()
    print(
        "❌ Attendance NOT marked."
    )


# ============================================================
# 15. SHOW ATTENDANCE
# ============================================================

print()
print(
    "============================================"
)
print(
    "ATTENDANCE RECORD"
)
print(
    "============================================"
)


if os.path.exists(
    ATTENDANCE_FILE
):

    attendance_df = pd.read_csv(
        ATTENDANCE_FILE
    )


    display(
        attendance_df
    )


else:

    print(
        "No attendance records found."
    )