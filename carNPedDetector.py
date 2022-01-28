import cv2

trainPedFileName = 'haarcascade_fullbody.xml'

trainCarFileName = 'carsTrainingData.xml'

videoFileName = 'jdm.mp4'

# This is the pretrained data for pedestrians loaded in with opencv
trainedPedData = cv2.CascadeClassifier(trainPedFileName)

# This pretrained car data being loaded in using opencv
trainedCarData = cv2.CascadeClassifier(trainCarFileName)

vehicleCamData = videoFileName

print("Detected cars and pedestrians succesfully....")