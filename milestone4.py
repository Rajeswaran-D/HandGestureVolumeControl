import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from math import hypot
import time

# ================== MEDIAPIPE SETUP ==================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
draw = mp.solutions.drawing_utils

# ================== CAMERA ==================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera not detected")
    exit()

# ================== VARIABLES ==================
prev_length = 0
last_time = 0
delay = 0.08

display_volume = 50
stability = 100

# ================== MAIN LOOP ==================
while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Resize for UI consistency
    frame = cv2.resize(frame, (400, h))

    # ================== UI BACKGROUND ==================
    ui = np.zeros((h, 700, 3), dtype=np.uint8)
    ui[:] = (25, 25, 25)

    # Place camera feed
    ui[:, :400] = frame

    # ================== HAND DETECTION ==================
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture = "None"
    distance_value = "--"

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            lm = []

            for id, point in enumerate(hand.landmark):
                x = int(point.x * 400)
                y = int(point.y * h)
                lm.append([id, x, y])

            if lm:
                x1, y1 = lm[4][1], lm[4][2]
                x2, y2 = lm[8][1], lm[8][2]

                length = hypot(x2 - x1, y2 - y1)
                distance_value = int(length)

                diff = length - prev_length
                current_time = time.time()

                if abs(diff) > 6 and current_time - last_time > delay:
                    steps = int(abs(diff) // 6)

                    if diff > 0:
                        gesture = "Hand Open"
                        for _ in range(steps):
                            pyautogui.press("volumeup")
                            display_volume = min(display_volume + 2, 100)
                    else:
                        gesture = "Pinch"
                        for _ in range(steps):
                            pyautogui.press("volumedown")
                            display_volume = max(display_volume - 2, 0)

                    stability = max(0, stability - 2)
                    prev_length = length
                    last_time = current_time
                else:
                    stability = min(100, stability + 0.4)

                # Draw gesture points
                cv2.circle(ui, (x1, y1), 8, (255, 0, 255), -1)
                cv2.circle(ui, (x2, y2), 8, (255, 0, 255), -1)
                cv2.line(ui, (x1, y1), (x2, y2), (255, 0, 255), 2)

                draw.draw_landmarks(
                    ui[:, :400],
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )

    # ================== UI PANEL ==================
    panel_x = 430

    cv2.putText(ui, "Enhanced Gesture UI",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.putText(ui, "Current Gesture:",
                (panel_x, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    cv2.putText(ui, gesture,
                (panel_x, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)

    cv2.putText(ui, f"Distance: {distance_value}",
                (panel_x, 165),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)

    # ================== VOLUME BAR ==================
    bar = np.interp(display_volume, [0, 100], [330, 210])

    cv2.rectangle(ui, (panel_x, 210), (panel_x + 40, 330),
                  (0, 255, 0), 2)
    cv2.rectangle(ui, (panel_x, int(bar)), (panel_x + 40, 330),
                  (0, 255, 0), -1)

    cv2.putText(ui, f"Volume: {display_volume}%",
                (panel_x - 10, 365),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    # ================== STABILITY ==================
    cv2.putText(ui, f"Stability: {int(stability)}%",
                (panel_x, 405),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 0, 255), 2)

    # ================== INSTRUCTIONS ==================
    cv2.putText(ui, "Instructions:",
                (20, h - 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 0), 2)

    cv2.putText(ui, "Pinch -> Decrease Volume",
                (20, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 1)

    cv2.putText(ui, "Open Hand -> Increase Volume",
                (20, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 1)

    # ================== SHOW ==================
    cv2.imshow("Enhanced Gesture UI & Feedback", ui)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================== CLEANUP ==================
cap.release()
cv2.destroyAllWindows()
