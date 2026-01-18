import cv2
import mediapipe as mp
import numpy as np
import math
import time
import pyautogui

# =========================
# LOAD WORD LIST
# =========================
with open("words.txt", "r", encoding="utf-8") as f:
    WORDS = [w.strip().lower() for w in f if w.strip()]

def predict(prefix, limit=3):
    if not prefix:
        return []
    return [w for w in WORDS if w.startswith(prefix)][:limit]

# =========================
# CONFIG (LOCKED BASELINE)
# =========================
PINCH_THRESHOLD = 40
ROTATION_GAIN = 1.0
SMOOTHING = 0.85

SHORT_PINCH_TIME = 0.25   # letter select
LONG_PINCH_TIME = 0.6    # prediction select

pyautogui.FAILSAFE = False

CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") + ["SPACE", "BKSP"]
N = len(CHARS)

# =========================
# HELPERS
# =========================
def angle(cx, cy, x, y):
    return math.atan2(y - cy, x - cx)

# =========================
# INIT
# =========================
cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(max_num_hands=1)

dial_angle = 0.0
prev_angle = None
smoothed_delta = 0.0

pinch_start_time = None
pinch_action = None

selected_index = 0
prediction_index = 0
typed_text = ""

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    pinch_action = None

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]

        ix = int(hand.landmark[8].x * w)
        iy = int(hand.landmark[8].y * h)

        mx = int(hand.landmark[12].x * w)
        my = int(hand.landmark[12].y * h)
        tx = int(hand.landmark[4].x * w)
        ty = int(hand.landmark[4].y * h)

        # ---------- ROTATION ----------
        current_angle = angle(cx, cy, ix, iy)
        if prev_angle is not None:
            delta = current_angle - prev_angle
            if delta > math.pi:
                delta -= 2 * math.pi
            if delta < -math.pi:
                delta += 2 * math.pi

            smoothed_delta = (
                SMOOTHING * smoothed_delta +
                (1 - SMOOTHING) * delta
            )
            dial_angle += smoothed_delta * ROTATION_GAIN

        prev_angle = current_angle

        # ---------- PINCH (TIME BASED) ----------
        dist = math.hypot(mx - tx, my - ty)

        if dist < PINCH_THRESHOLD:
            if pinch_start_time is None:
                pinch_start_time = time.time()
        else:
            if pinch_start_time is not None:
                duration = time.time() - pinch_start_time
                if duration < SHORT_PINCH_TIME:
                    pinch_action = "letter"
                elif duration >= LONG_PINCH_TIME:
                    pinch_action = "prediction"
                pinch_start_time = None

    # =========================
    # SELECTION
    # =========================
    selected_index = int(
        round(((-dial_angle) % (2 * math.pi)) / (2 * math.pi) * N)
    ) % N

    ch = CHARS[selected_index]

    prefix = typed_text.split(" ")[-1]
    preds = predict(prefix)

    if preds:
        prediction_index = selected_index % len(preds)

    # =========================
    # INPUT HANDLING
    # =========================
    if pinch_action == "letter":
        if ch == "SPACE":
            typed_text += " "
            pyautogui.press("space")
        elif ch == "BKSP":
            typed_text = typed_text[:-1]
            pyautogui.press("backspace")
        else:
            typed_text += ch.lower()
            pyautogui.write(ch.lower())

    elif pinch_action == "prediction" and preds:
        chosen = preds[prediction_index]

        # ---- DELETE PREFIX FROM ACTIVE APP ----
        current_prefix = typed_text.split(" ")[-1]
        for _ in range(len(current_prefix)):
            pyautogui.press("backspace")

        # ---- REPLACE WITH FULL WORD ----
        words = typed_text.split(" ")
        typed_text = " ".join(words[:-1] + [chosen]) + " "
        pyautogui.write(chosen + " ")

    # =========================
    # CAMERA WINDOW
    # =========================
    cv2.imshow("Camera", frame)

    # =========================
    # KEYBOARD + PREDICTION UI
    # =========================
    size = 600
    center = size // 2
    radius = 220

    ui = np.zeros((size, size, 3), dtype=np.uint8)
    ui[:] = (20, 20, 20)

    for i, c in enumerate(CHARS):
        theta = 2 * math.pi * i / N - math.pi / 2
        x = int(center + radius * math.cos(theta))
        y = int(center + radius * math.sin(theta))

        if i == selected_index:
            cv2.circle(ui, (x, y), 28, (0, 220, 255), -1)
            color = (15, 15, 15)
            scale = 0.9
        else:
            color = (220, 220, 220)
            scale = 0.6

        ts = cv2.getTextSize(c, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
        cv2.putText(
            ui,
            c,
            (x - ts[0] // 2, y + ts[1] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            2,
            cv2.LINE_AA,
        )

    # ---------- PREDICTIONS ----------
    y = 500
    for i, p in enumerate(preds):
        color = (0, 255, 255) if i == prediction_index else (180, 180, 180)
        cv2.putText(
            ui,
            f"{i+1}. {p}",
            (180, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
        y += 30

    # ---------- HINT ----------
    if pinch_start_time:
        cv2.putText(
            ui,
            "Hold pinch for prediction",
            (140, 470),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    cv2.imshow("Circular Keyboard", ui)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
