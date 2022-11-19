import cv2 as cv


def nothing(Null):
    pass


cv.namedWindow("Camera")
cv.createTrackbar("exposure", "Camera", 1, 7, nothing)

capture = cv.VideoCapture(0)
i = 0
while True:
    exposure = cv.getTrackbarPos("exposure", "Camera")
    capture.set(15, -exposure)
    _, frame = capture.read()
    frame = cv.flip(frame, 1)

    cv.imshow("Main", frame)
    keyPressed = cv.waitKey(1) & 0xFF
    if keyPressed == ord("c"):
        cv.imwrite(f"photos/{i}.png", frame)
        print(f"Captured {i} Done")
        i += 1
    elif keyPressed == ord(" "):
        break
