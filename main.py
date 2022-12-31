import cv2
import os
import pickle
import face_recognition
import numpy as np
import cvzone
from datetime import datetime

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://faceattendancerealtime-8230c-default-rtdb.firebaseio.com/",
    'storageBucket': "faceattendancerealtime-8230c.appspot.com"
})

bucket = storage.bucket()

# Take a capture from the local camera
cap = cv2.VideoCapture(0)

# Image Capture Dimensions
cap.set(3,640)
cap.set(4,480)

imgBackground = cv2.imread('Resources/background.png')

# Importing the mode images into the list
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

# print(len(imgModeList))

# Load the encoding files
file = open('EncodeFile.p','rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
print(studentIds)

modeType = 0
counter = 0
id = -1
imgStudent = []

while True:
    success, img = cap.read()

    # Minimizing the computation power by making the images smaller only for face recognition, not the webcam frame
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    # create the webcam video inside the imgBackground / Mask object
    # 162, 162+480 = height start & end point, 55, 55+640 = width start & end point
    imgBackground[162:162 + 480, 55:55 + 640] = img
    
    # create the mode images inside the imgBackground / Mask object
    # 44, 44+633 = height start & end point, 808, 808+414 = width start & end point
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    # if face is detected
    if faceCurFrame:
        # Looping 2 lists at once using zip method
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            
            # the lower faceDis, the bigger possibilities that 2 images are same
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            # print("matches", matches)
            # print("faceDis", faceDis)

            matchIndex = np.argmin(faceDis)
            # print("MatchIndex", matchIndex)

            if matches[matchIndex]:
                # print("Known Face Detected")
                # print(studentIds[matchIndex])

                # Create a bounding box start from the webcam frame
                y1, x2, y2, x1 = faceLoc

                # Multiplied 4 because of the previous resizing
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4 
                bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1

                # Bounding box in cvzone are more vancy than openCV
                imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
                id = studentIds[matchIndex]

                if counter == 0:
                    cvzone.putTextRect(imgBackground, "Loading", (275,400))
                    cv2.imshow("Face Attendance", imgBackground)
                    cv2.waitKey(1)
                    counter = 1
                    modeType = 1

        if counter != 0:
            if counter == 1:
                # Get the Data from firebase db
                studentInfo = db.reference(f'Students/{id}').get()
                print(studentInfo)

                # Get the Image from firebase storage
                blob = bucket.get_blob(f'Images/{id}.png')

                # download image as a string and decode it
                array = np.frombuffer(blob.download_as_string(), np.uint8)
                imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)

                # Update data of attendance
                datetimeObject = datetime.strptime(studentInfo['last_attendance_time'],
                                    "%Y-%m-%d %H:%M:%S")
                secondsElapsed = (datetime.now() - datetimeObject).total_seconds() # now - timeLastAttendance
                print(secondsElapsed)

                # If now - timeLastAttendance is greater than 30 seconds
                if secondsElapsed > 30:
                    ref = db.reference(f'Students/{id}')
                    studentInfo['total_attendance'] += 1
                    ref.child('total_attendance').set(studentInfo['total_attendance'])
                    ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                else: 
                    modeType = 3
                    counter = 0
                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
            
            if modeType != 3:

                if 10 < counter < 20:
                    modeType = 2
                
                imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                if counter <= 10:
                    # Put the text from firebase db in the Image Background
                    cv2.putText(imgBackground, str(studentInfo['total_attendance']), (861, 125),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(studentInfo['major']), (1006, 550),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(id), (1006, 493),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(studentInfo['standing']), (910, 625),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                    cv2.putText(imgBackground, str(studentInfo['year']), (1025, 625),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                    cv2.putText(imgBackground, str(studentInfo['starting_year']), (1125, 625),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

                    # Put the text from firebase db centered in the Image Background
                    (w, h), _ = cv2.getTextSize(studentInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                    offset = (414 - w) // 2 # left distance to center
                    cv2.putText(imgBackground, str(studentInfo['name']), (808 + offset, 445),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

                    # Put the image student in the image background
                    imgBackground[175:175 + 216, 909:909 + 216] = imgStudent

                counter += 1

                if counter >= 20:
                    counter = 0
                    modeType = 0
                    studentInfo = []
                    imgStudent = []
                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
    else:
        modeType = 0
        counter = 0

    # cv2.imshow("Webcam", img) # show window of your camera
    cv2.imshow("Face Attendance", imgBackground)
    if cv2.waitKey(1) & 0xFF == ord('q'): # press q To exit
        break

cap.release()
cv2.destroyAllWindows()