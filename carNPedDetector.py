import cv2

trainPedFileName = 'haarcascade_fullbody.xml'

trainCarFileName = 'carsTrainingData.xml'

videoFileName = 'jdm.mp4'

# This is the pretrained data for pedestrians loaded in with opencv
trainedPedData = cv2.CascadeClassifier(trainPedFileName)

# This pretrained car data being loaded in using opencv
trainedCarData = cv2.CascadeClassifier(trainCarFileName)

# Loaded in video file for to act as cehicle camera
vehicleCamData = cv2.VideoCapture(videoFileName)

# Looping through frames for camera
while True:

    # Take in current frame
    readSuccess, captFrame = vehicleCamData.read()

    # Checks if video input worked
    if readSuccess:
        print("Nice Ready for GrayScale")

    else:
        break

    # Shows video
    cv2.imshow('Car and Pedestrian Detector', captFrame)

    # Time to wait for each frame in video
    key = cv2.waitKey(1)

    # Can end program by pressing b
    if key == 98:
        break

# Clears mem
vehicleCamData.release()

print("Program Compiled")