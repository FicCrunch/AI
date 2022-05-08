import cv2
from random import randint

# Coordinates of the detected faces
#  [[ 75 286  76  76]
#   [377 278  70  70]
#   [539 278  85  85]]


def video():
    # train application with pre-trained images
    trained_face_data = cv2.CascadeClassifier(
        "data/haarcascades_cuda/haarcascade_frontalface_default.xml"
    )

    # choose an image to detect faces in
    # img = cv2.imread('0-8.jpg')
    webcam = cv2.VideoCapture(0)

    while True:
        # read the current frame
        successful_frame_read, frame = webcam.read()

        # Must convert to grayscale
        # grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_coordinates = trained_face_data.detectMultiScale(frame)
        for (x, y, w, h) in face_coordinates:
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (randint(0, 256), randint(0, 256), randint(0, 256)),
                2,
            )

        # show it
        cv2.imshow("Clever Programmer", frame)
        cv2.waitKey(1)


def image():
    # train application with pre-trained images
    trained_face_data = cv2.CascadeClassifier(
        "data/haarcascades_cuda/haarcascade_frontalface_default.xml"
    )

    # choose an image to detect faces in
    img = cv2.imread("0-8.jpg")

    # Must convert to grayscale
    grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect face then returns coordinates
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    # print(face_coordinates)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (randint(0, 256), randint(0, 256), randint(0, 256)),
            15,
        )

    # show it
    cv2.imshow("Clever Programmer", img)
    cv2.waitKey()

    print("Code Completed")


def video_pedestrian():
    video = cv2.VideoCapture("Pedestrian.mp4")

    pedestrian_tracker = cv2.CascadeClassifier(
        "data/haarcascades/haarcascade_fullbody.xml"
    )

    while True:
        (read_successful, frame) = video.read()

        if read_successful:
            grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            break

        people = pedestrian_tracker.detectMultiScale(grayscaled_frame, 240)
        for (x, y, w, h) in people:
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (randint(0, 256), randint(0, 256), randint(0, 256)),
                2,
            )

        cv2.imshow("Car Tracker", frame)
        cv2.waitKey(1)


# image()
# video()
video_pedestrian()
