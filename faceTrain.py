import os
import pickle

import cv2 as cv
import numpy as np
from PIL import Image

faceCascade = cv.CascadeClassifier("data/haarcascade_frontalface_default.xml")
recognizer = cv.face.LBPHFaceRecognizer_create()

baseDir = os.path.dirname(os.path.abspath(__file__))
photosDir = os.path.join(baseDir, "photos")

currentId = 0
labelIds = {}
yLabel = []
xTrain = []

for root, dirs, files in os.walk(photosDir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "_").lower()
            print(label, path)
            if not label in labelIds:
                labelIds[label] = currentId
                currentId += 1
            id_ = labelIds[label]
            print(labelIds)
            # yLabel.append(label)
            # xTrain.append(path)
            pilPhoto = Image.open(path).convert("L")
            size = (550, 550)
            finalPhoto = pilPhoto.resize(size,Image.ANTIALIAS)
            photoArray = np.array(finalPhoto, "uint8")
            print(photoArray)
            faces = faceCascade.detectMultiScale(photoArray, scaleFactor=1.1, minNeighbors=2)
            for (x, y, w, h) in faces:
                print(x, y, w, h)
                roi = photoArray[y:y + h, x:x + w]
                xTrain.append(roi)
                yLabel.append(id_)
# print(yLabel)
# print(xTrain)

with open("labels.pickle", "wb") as f:
    pickle.dump(labelIds, f)

recognizer.train(xTrain, np.array(yLabel))
recognizer.save("trainer.yml")
