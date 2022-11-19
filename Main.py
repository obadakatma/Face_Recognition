import pickle
import time

import cv2 as cv


def nothing(Null):
    pass


faceCascade = cv.CascadeClassifier("data/haarcascade_frontalface_default.xml")
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {"person_name": 1}
with open("labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

cv.namedWindow("Camera")
cv.createTrackbar("exposure", "Camera", 1, 7, nothing)

capture = cv.VideoCapture(0)

i = 0
pTime, cTime = 0, 0
while True:
    exposure = cv.getTrackbarPos("exposure", "Camera")
    capture.set(15, -exposure)
    _, frame = capture.read()
    frame = cv.flip(frame, 1)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)
    for (x, y, w, h) in faces:
        # print(x, y, w, h)
        roiGray = gray[y:y + h, x:x + w]
        roiColor = frame[y:y + h, x:x + w]

        id_, conf = recognizer.predict(roiGray)
        if 30 <= conf <= 85:
            print(id_)
            print(labels[id_])
            cv.putText(frame, labels[id_], (x, y - 4), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(frame, str(int(fps)), (10, 35), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    cv.imshow("Main", frame)
    if cv.waitKey(1) & 0xFF == ord(" "):
        break
