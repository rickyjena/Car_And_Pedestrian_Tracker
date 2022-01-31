import cv2

trainPedFileName = 'haarcascade_fullbody.xml'

trainCarFileName = 'carsTrainingData.xml'

videoFileName = 'tokyoStreets.mp4'

# This is the pretrained data for pedestrians loaded in with opencv
trainedPedData = cv2.CascadeClassifier(trainPedFileName)

# This pretrained car data being loaded in using opencv
trainedCarData = cv2.CascadeClassifier(trainCarFileName)

# Loaded in video file for to act as vehicle camera
vehicleCamData = cv2.VideoCapture(videoFileName)

# Looping through frames for camera
while True:

    # Take in current frame
    readSuccess, captFrame = vehicleCamData.read()

    # Checks if video input worked
    if readSuccess:
        #grayscales the video
        grayscaleVid = cv2.cvtColor(captFrame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # Get coords of pedestrians in video
    pedCoords = trainedPedData.detectMultiScale(grayscaleVid)

    #Creates rectangle for pedestrians
    for x in pedCoords:
        (startX, startY, width, height) = x
        startPoint = (startX, startY)
        endPoint = (startX + width, startY + height)

        # Set box color to red
        boxColor = (0, 0, 255)

        # Box side width
        boxWidth = 2

        # Draw rectangles around the pedestrian using coords
        cv2.rectangle(captFrame, startPoint, endPoint, boxColor, boxWidth)

    # Get coordinates of cars in video
    carCoords = trainedCarData.detectMultiScale(grayscaleVid)

    #Creates rectangle for cars
    for x in carCoords:
        (startX, startY, width, height) = x
        startPoint = (startX, startY)
        endPoint = (startX + width, startY + height)

        # Set box color to green
        boxColor = (0, 255, 0)

        # Box side width
        boxWidth = 2

        # Draw rectangles around the cars using coords
        cv2.rectangle(captFrame, startPoint, endPoint, boxColor, boxWidth)

    # Shows  grayscale video
    cv2.imshow('Car and Pedestrian Detector', captFrame)

    # Time to wait for each frame in video
    key = cv2.waitKey(1)

    # Can end program by pressing b
    if key == 98:
        break

# Clears mem
vehicleCamData.release()

print("Program Compiled")