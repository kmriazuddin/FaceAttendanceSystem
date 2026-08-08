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

# ============================================================
# 6. LOAD AND ENCODE KNOWN FACES
# ============================================================

def load_known_faces(folder):

    known_encodings = []
    known_names = []

    print()
    print("============================================")
    print("ENCODING KNOWN FACES")
    print("============================================")
    print()


    for filename in sorted(
        os.listdir(folder)
    ):

        if not filename.lower().endswith(
            ('.jpg', '.jpeg', '.png')
        ):

            continue


        path = os.path.join(
            folder,
            filename
        )


        try:

            # Load image
            image = face_recognition.load_image_file(
                path
            )


            # Detect faces
            face_locations = (
                face_recognition.face_locations(
                    image
                )
            )


            # No face
            if len(face_locations) == 0:

                print(
                    f"❌ No face found: {filename}"
                )

                continue


            # Multiple faces
            if len(face_locations) > 1:

                print(
                    f"❌ Multiple faces found: {filename}"
                )

                print(
                    "   Use an image with ONE person."
                )

                continue


            # Face encoding
            face_encoding = (
                face_recognition.face_encodings(
                    image,
                    face_locations
                )[0]
            )


            # Student name
            name = os.path.splitext(
                filename
            )[0]


            known_encodings.append(
                face_encoding
            )

            known_names.append(
                name
            )


            print(
                f"✅ Encoded: {name}"
            )


        except Exception as e:

            print(
                f"❌ Error processing: {filename}"
            )

            print(
                "   Error:",
                e
            )


    return (
        known_encodings,
        known_names
    )


known_encodings, known_names = (
    load_known_faces(
        KNOWN_FACES_DIR
    )
)


print()
print("============================================")
print(
    "TOTAL VALID USERS:",
    len(known_names)
)
print("============================================")


for name in known_names:

    print(
        "•",
        name
    )


# ============================================================
# 7. VERIFICATION SETTINGS
# ============================================================

FACE_TOLERANCE = 0.48

# Same person must be recognized in
# at least this many frames.
MIN_RECOGNITION_COUNT = 5

# Required horizontal head movement.
MIN_HEAD_MOVEMENT = 40

# Maximum verification time.
VERIFICATION_TIMEOUT = 15


print()
print("============================================")
print("VERIFICATION SETTINGS")
print("============================================")

print(
    "Face tolerance:",
    FACE_TOLERANCE
)

print(
    "Minimum recognition count:",
    MIN_RECOGNITION_COUNT
)

print(
    "Minimum head movement:",
    MIN_HEAD_MOVEMENT
)

print(
    "Verification timeout:",
    VERIFICATION_TIMEOUT,
    "seconds"
)


# ============================================================
# 8. BROWSER WEBCAM VIDEO CAPTURE
# ============================================================

def capture_webcam_video(
    filename='verification.webm',
    duration=8
):


    javascript_code = f"""
    async function recordFaceVerification() {{

        const container =
            document.createElement('div');


        // Title
        const title =
            document.createElement('h3');

        title.innerText =
            'Face Verification';


        container.appendChild(
            title
        );


        // Instructions
        const instruction =
            document.createElement('p');

        instruction.innerText =
            'Look at the camera and slowly move your head LEFT and RIGHT.';


        container.appendChild(
            instruction
        );


        // Video
        const video =
            document.createElement('video');


        video.style.width =
            '640px';

        video.style.height =
            '480px';

        video.autoplay = true;


        container.appendChild(
            video
        );


        // Start button
        const startButton =
            document.createElement('button');


        startButton.innerText =
            'Start Verification';


        startButton.style.fontSize =
            '18px';


        startButton.style.padding =
            '10px';


        container.appendChild(
            startButton
        );


        document.body.appendChild(
            container
        );


        // Camera permission
        const stream =
            await navigator.mediaDevices.getUserMedia({{
                video: {{
                    width: {{ideal: 640}},
                    height: {{ideal: 480}}
                }},
                audio: false
            }});


        video.srcObject =
            stream;


        await video.play();


        // Wait for Start button
        await new Promise(
            resolve =>
                startButton.onclick =
                    resolve
        );


        startButton.disabled =
            true;


        startButton.innerText =
            'Verification Running...';


        // Media recorder
        const recorder =
            new MediaRecorder(
                stream
            );


        let chunks = [];


        recorder.ondataavailable =
            function(event) {{

                if (
                    event.data.size > 0
                ) {{

                    chunks.push(
                        event.data
                    );

                }}

            }};


        recorder.start();


        // Record
        await new Promise(
            resolve =>
                setTimeout(
                    resolve,
                    {duration * 1000}
                )
        );


        recorder.stop();


        await new Promise(
            resolve =>
                recorder.onstop =
                    resolve
        );


        // Stop camera
        stream
            .getTracks()
            .forEach(
                track =>
                    track.stop()
            );


        // Remove UI
        container.remove();


        // Create video blob
        const blob =
            new Blob(
                chunks,
                {{
                    type:
                        'video/webm'
                }}
            );


        // Convert to Base64
        const reader =
            new FileReader();


        return await new Promise(
            resolve => {{

                reader.onloadend =
                    function() {{

                        resolve(
                            reader.result
                        );

                    }};


                reader.readAsDataURL(
                    blob
                );

            }}
        );

    }}


    recordFaceVerification();
    """


    display(
        Javascript(
            javascript_code
        )
    )


    # Get video from browser
    data = eval_js(
        "recordFaceVerification()"
    )


    # Decode Base64
    binary = base64.b64decode(
        data.split(',')[1]
    )


    # Save video
    with open(
        filename,
        'wb'
    ) as f:

        f.write(
            binary
        )


    return filename


print()
print(
    "✅ Webcam capture function ready."
)


# ============================================================
# 9. FACE VERIFICATION FOR A SINGLE FRAME
# ============================================================

def verify_face_frame(
    frame,
    known_encodings,
    known_names
):


    # Convert BGR -> RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # Detect faces
    face_locations = (
        face_recognition.face_locations(
            rgb
        )
    )


    # No face
    if len(face_locations) == 0:

        return (
            None,
            None,
            None
        )


    # Multiple faces
    if len(face_locations) > 1:

        return (
            "MULTIPLE",
            None,
            None
        )


    # Encode detected face
    encodings = (
        face_recognition.face_encodings(
            rgb,
            face_locations
        )
    )


    if len(encodings) == 0:

        return (
            None,
            None,
            None
        )


    face_encoding = encodings[0]


    # Compare with known faces
    distances = (
        face_recognition.face_distance(
            known_encodings,
            face_encoding
        )
    )


    if len(distances) == 0:

        return (
            "UNKNOWN",
            None,
            face_locations[0]
        )


    # Find closest face
    best_index = np.argmin(
        distances
    )


    best_distance = distances[
        best_index
    ]


    # Face matched
    if best_distance <= FACE_TOLERANCE:

        name = known_names[
            best_index
        ]


        return (
            name,
            best_distance,
            face_locations[0]
        )


    # Face not matched
    return (
        "UNKNOWN",
        best_distance,
        face_locations[0]
    )