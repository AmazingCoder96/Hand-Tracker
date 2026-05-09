import numpy as np 
import cv2
import mediapipe as mp
import pyautogui
import time

model_path = 'hand_landmarker.task' 

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

screen_w, screen_h = pyautogui.size()

pyautogui.FAILSAFE = True

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

prev_x, prev_y = 0, 0
smoothing = 5 

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        timestamp = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp)

        if result.hand_landmarks:
            for i, landmarks in enumerate(result.hand_landmarks):
                hand_label = result.handedness[i][0].category_name
                
                if hand_label == "Right":
                    index_tip = landmarks[8]
                    
                    ix = index_tip.x * w
                    iy = index_tip.y * h

                    margin = 100
                    screen_x = np.interp(ix, (margin, w - margin), (0, screen_w))
                    screen_y = np.interp(iy, (margin, h - margin), (0, screen_h))

                    curr_x = prev_x + (screen_x - prev_x) / smoothing
                    curr_y = prev_y + (screen_y - prev_y) / smoothing

                    pyautogui.moveTo(curr_x, curr_y, _pause=False)
                    
                    prev_x, prev_y = curr_x, curr_y

                    cv2.circle(frame, (int(ix), int(iy)), 5, (0, 255, 255), -1)

        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
