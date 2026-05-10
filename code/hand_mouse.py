import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

model_path = 'hand_landmarker.task'
sensitivity_factor = 5
smoothing = 8
prev_x, prev_y = 0, 0
screen_w, screen_h = pyautogui.size()

mouse_is_down = False

click_modes = ['left', 'right', 'middle']
current_mode_index = 0
left_was_in_center = False

window_name = "Hand Mouse Pic-in-Pic"
taskbar_height = 80
pip_width = 426
pip_height = 240
pip_x = screen_w - pip_width
pip_y = screen_h - pip_height - taskbar_height

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

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

        left_hand_present = False

        zone_x_min, zone_x_max = 0.4, 0.7
        zone_y_min, zone_y_max = 0.35, 0.65

        zone_color = (0, 255, 0) if left_was_in_center else (255, 0, 0)
        cv2.rectangle(frame, (int(w * zone_x_min), int(h * zone_y_min)),
                      (int(w * zone_x_max), int(h * zone_y_max)), zone_color, 2)
        cv2.putText(frame, "TOGGLE ZONE", (int(w * zone_x_min) + 5, int(h * zone_y_min) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone_color, 1)

        if result.hand_landmarks:
            for i, landmarks in enumerate(result.hand_landmarks):
                hand_label = result.handedness[i][0].category_name

                if hand_label == "Left":
                    left_hand_present = True

                    wrist_x = landmarks[0].x
                    wrist_y = landmarks[0].y

                    is_in_center = (zone_x_min < wrist_x < zone_x_max) and (zone_y_min < wrist_y < zone_y_max)

                    if is_in_center and not left_was_in_center:
                        left_was_in_center = True
                    elif not is_in_center and left_was_in_center:
                        current_mode_index = (current_mode_index + 1) % 3
                        left_was_in_center = False
                        print(f"Mode switched to: {click_modes[current_mode_index]}")

                    is_fist_now = (landmarks[8].y > landmarks[6].y and
                                   landmarks[12].y > landmarks[10].y and
                                   landmarks[16].y > landmarks[14].y and
                                   landmarks[20].y > landmarks[18].y)

                    current_button = click_modes[current_mode_index]

                    if is_fist_now and not mouse_is_down:
                        pyautogui.mouseDown(button=current_button)
                        mouse_is_down = True
                    elif not is_fist_now and mouse_is_down:
                        pyautogui.mouseUp(button=current_button)
                        mouse_is_down = False

                if hand_label == "Right":
                    index_tip = landmarks[8]
                    ix, iy = index_tip.x * w, index_tip.y * h

                    x_min, x_max = w / 2 - (w / (2 * sensitivity_factor)), w / 2 + (w / (2 * sensitivity_factor))
                    y_min, y_max = h / 2 - (h / (2 * sensitivity_factor)), h / 2 + (h / (2 * sensitivity_factor))

                    screen_x = np.interp(ix, (x_min, x_max), (0, screen_w))
                    screen_y = np.interp(iy, (y_min, y_max), (0, screen_h))

                    curr_x = prev_x + (np.clip(screen_x, 0, screen_w) - prev_x) / smoothing
                    curr_y = prev_y + (np.clip(screen_y, 0, screen_h) - prev_y) / smoothing

                    pyautogui.moveTo(curr_x, curr_y, _pause=False)
                    prev_x, prev_y = curr_x, curr_y

                    color = (0, 0, 255) if mouse_is_down else (0, 255, 255)
                    cv2.circle(frame, (int(ix), int(iy)), 15, color, -1)
                    cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 255), 2)

        if not left_hand_present and mouse_is_down:
            pyautogui.mouseUp(button=click_modes[current_mode_index])
            mouse_is_down = False

        mode_text = f"Mode: {click_modes[current_mode_index].upper()} CLICK"
        cv2.putText(frame, mode_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        pip_frame = cv2.resize(frame, (pip_width, pip_height))

        cv2.imshow(window_name, pip_frame)
        cv2.resizeWindow(window_name, pip_width, pip_height)
        cv2.moveWindow(window_name, pip_x, pip_y)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
