import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

cap = cv2.VideoCapture(0)

# cooldown to avoid rapid repeats
cooldown_s = 0.35
last_cmd_t = 0.0
last_cmd = None

TIP_IDS = [4, 8, 12, 16, 20]
PIP_IDS = [2, 6, 10, 14, 18]
MCP_IDS = [1, 5, 9, 13, 17]

def count_fingers(lm, handedness_label: str):
    fingers = 0

    # ---- Thumb (optional: disable if thumb causes confusion) ----
    # If you want thumb ignored completely, comment this block.
    if handedness_label == "Right":
        if (lm[TIP_IDS[0]].x - lm[PIP_IDS[0]].x) > 0.04:
            fingers += 1
    else:
        if (lm[PIP_IDS[0]].x - lm[TIP_IDS[0]].x) > 0.04:
            fingers += 1

    # ---- Other 4 fingers (strict + threshold) ----
    for i in range(1, 5):
        tip = lm[TIP_IDS[i]]
        pip = lm[PIP_IDS[i]]
        mcp = lm[MCP_IDS[i]]

        # y axis: smaller y = higher on screen
        is_above = (tip.y < pip.y) and (tip.y < mcp.y)
        strong_enough = (pip.y - tip.y) > 0.03  # threshold; try 0.02-0.05

        if is_above and strong_enough:
            fingers += 1

    return fingers


def fingers_to_command(fcnt: int):
    # You asked: 1=up, 2=down; adding left/right too
    if fcnt == 1:
        return "UP"
    if fcnt == 2:
        return "DOWN"
    if fcnt == 3:
        return "LEFT"
    if fcnt == 4:
        return "RIGHT"
    if fcnt == 0:
        return "STOP"
    return None  # 5 fingers = ignore (or you can set some action)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    cmd = None
    fcnt = None

    if res.multi_hand_landmarks:
        hand_lm = res.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

        # handedness label ("Left"/"Right")
        handedness_label = "Right"
        if res.multi_handedness and len(res.multi_handedness) > 0:
            handedness_label = res.multi_handedness[0].classification[0].label

        lm = hand_lm.landmark
        fcnt = count_fingers(lm, handedness_label)
        cmd = fingers_to_command(fcnt)

        # display
        cv2.putText(frame, f"Hand: {handedness_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, f"Fingers: {fcnt}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, f"Cmd: {cmd}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # cooldown print (so it doesn't spam)
        now = time.time()
        if cmd and (now - last_cmd_t) > cooldown_s and cmd != last_cmd:
            print(cmd)
            last_cmd = cmd
            last_cmd_t = now

    cv2.imshow("Finger Control (press q)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
