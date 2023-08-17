import cv2
import mediapipe as mp
import numpy as np
from google.protobuf.json_format import MessageToDict
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc

# Initializing the Hand Tracking Model
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2)

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

# Initialize brightness control variables
minBrightness = 0
maxBrightness = 100

# Initialize volume control variables
minVolume = -65
maxVolume = 0

# Initialize audio control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

while True:
    # Read video frame by frame
    success, img = cap.read()

    # Flip the image(frame)
    img = cv2.flip(img, 1)

    # Convert BGR image to RGB image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the RGB image
    results = hands.process(imgRGB)

    # If hands are present in the image(frame)
    if results.multi_hand_landmarks:

        # Loop through each detected hand
        for handLandmarks in results.multi_hand_landmarks:
            landmarkList = []

            for idx, landmark in enumerate(handLandmarks.landmark):
                height, width, _ = img.shape
                x, y = int(landmark.x * width), int(landmark.y * height)
                landmarkList.append([idx, x, y])

            # Check if it's the left hand
            if MessageToDict(results.multi_handedness[0])['classification'][0]['label'] == 'Left':
                # Calculate the distance between thumb and index finger
                x1, y1 = landmarkList[4][1], landmarkList[4][2]
                x2, y2 = landmarkList[8][1], landmarkList[8][2]
                length = np.hypot(x2 - x1, y2 - y1)

                # Map the distance to brightness level
                brightness = np.interp(length, [15, 220], [minBrightness, maxBrightness])
                sbc.set_brightness(int(brightness))

            # Check if it's the right hand
            if MessageToDict(results.multi_handedness[0])['classification'][0]['label'] == 'Right':
                # Calculate the distance between thumb and index finger
                x1, y1 = landmarkList[4][1], landmarkList[4][2]
                x2, y2 = landmarkList[8][1], landmarkList[8][2]
                length = np.hypot(x2 - x1, y2 - y1)

                # Map the distance to volume level
                volume_level = np.interp(length, [15, 220], [minVolume, maxVolume])
                volume.SetMasterVolumeLevel(volume_level, None)

    # Display the image with hand labels
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
