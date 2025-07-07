import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math

# Initialization
# Mode constants
EYE_TRACKER_MODE = 0
HAND_GESTURE_MODE = 1

# UI & Control Constants 
# Layout - buttons and their positions
BUTTON_RADIUS = 60
BUTTON_Y = 80
SWITCH_BUTTON_X = 370
COLOR_BUTTON_X = 510
EXIT_BUTTON_X = 650
CLEAR_BUTTON_X = 790 
CLICK_RADIUS = 70

# Colors
BUTTON_COLOR = (100, 100, 100)
BUTTON_TEXT_COLOR = (255, 255, 255)
HIGHLIGHT_COLOR = (0, 255, 0)
CANVAS_BG = np.zeros((720, 1280, 3), np.uint8)

# Drawing & Color Palette
COLOR_PALETTE = [
    (0, 255, 0),      # Green
    (255, 0, 0),      # Blue
    (0, 0, 255),      # Red
    (255, 255, 0),    # Cyan
    (42, 42, 165),    # Brown
    (255, 0, 255),    # Magenta
    (128, 0, 128),    # Purple
    (128, 128, 128),  # Grey
    (0, 0, 0)         # Rainbow
]
COLOR_NAMES = ["Green", "Blue", "Red", "Sky Blue", "Brown", "Pink", "Purple", "Grey", "Rainbow ✨"]
ERASER_COLOR = (0, 0, 0)
BRUSH_THICKNESS = 15
ERASER_THICKNESS = 50

# --- State & Timing Variables ---
current_mode = EYE_TRACKER_MODE
current_color_index = 0
DRAW_COLOR = COLOR_PALETTE[current_color_index]
rainbow_hue = 0
xp, yp = 0, 0
mode_message = "Eye Tracker Enabled"
ACTION_COOLDOWN = 0.5
last_action_time = 0

# Eye Tracking & Mouse Control 
EYE_AR_THRESH = 0.012
SMOOTHING_FACTOR = 0.1
MOVEMENT_AMPLIFICATION = 3.0
prev_x, prev_y = 0, 0
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# Initializing camera
cam = cv2.VideoCapture(0)
cam.set(3, 1280)
cam.set(4, 720)

# Initializing MediaPipe
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Landmark constants for clarity
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145
IRIS_TRACKING_POINTS = range(474, 478)
INDEX_FINGER_TIP = 8
THUMB_TIP = 4

# Get screen dimensions
screen_w, screen_h = pyautogui.size()
canvas = CANVAS_BG.copy()
WINDOW_NAME = 'AI Virtual Control'

def draw_round_button(frame, center_x, center_y, text, highlight=False):
    """Draws a round button, highlighting it if active."""
    color = HIGHLIGHT_COLOR if highlight else BUTTON_COLOR
    cv2.circle(frame, (center_x, center_y), BUTTON_RADIUS, color, -1)
    cv2.circle(frame, (center_x, center_y), BUTTON_RADIUS, (255, 255, 255), 3)
    
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = center_x - text_size[0] // 2
    text_y = center_y + text_size[1] // 2
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BUTTON_TEXT_COLOR, 2)

def is_point_in_button(x, y, center_x, center_y):
    """Checks if a point is within the button's clickable radius."""
    return math.sqrt((x - center_x)**2 + (y - center_y)**2) <= CLICK_RADIUS

def change_color():
    """Cycles to the next color in the palette."""
    global current_color_index, DRAW_COLOR
    current_color_index = (current_color_index + 1) % len(COLOR_PALETTE)
    DRAW_COLOR = COLOR_PALETTE[current_color_index]
    return COLOR_NAMES[current_color_index]

def get_current_draw_color():
    """Gets the current drawing color, handling the rainbow effect dynamically."""
    global rainbow_hue
    if COLOR_NAMES[current_color_index] == "Rainbow ✨":
        rainbow_hue = (rainbow_hue + 5) % 180
        hsv_color = np.array([[[rainbow_hue, 255, 255]]], dtype=np.uint8)
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        return tuple(map(int, bgr_color))
    else:
        return DRAW_COLOR

def handle_eye_tracking(frame, landmark_points):
    """Handles all eye tracking logic: cursor movement and blinking."""
    global prev_x, prev_y
    if not landmark_points:
        return None

    landmarks = landmark_points[0].landmark
    frame_h, frame_w, _ = frame.shape

    iris_pos = landmarks[IRIS_TRACKING_POINTS.start]
    screen_x = screen_w * iris_pos.x
    screen_y = screen_h * iris_pos.y
    
    center_x, center_y = screen_w / 2, screen_h / 2
    offset_x = (screen_x - center_x) * MOVEMENT_AMPLIFICATION
    offset_y = (screen_y - center_y) * MOVEMENT_AMPLIFICATION
    amplified_x = max(0, min(screen_w - 1, center_x + offset_x))
    amplified_y = max(0, min(screen_h - 1, center_y + offset_y))

    smooth_x = prev_x * SMOOTHING_FACTOR + amplified_x * (1 - SMOOTHING_FACTOR)
    smooth_y = prev_y * SMOOTHING_FACTOR + amplified_y * (1 - SMOOTHING_FACTOR)

    pyautogui.moveTo(smooth_x, smooth_y, duration=0)
    prev_x, prev_y = smooth_x, smooth_y

    p_top = landmarks[RIGHT_EYE_TOP]
    p_bottom = landmarks[RIGHT_EYE_BOTTOM]
    
    p_top_coord = (int(p_top.x * frame_w), int(p_top.y * frame_h))
    p_bottom_coord = (int(p_bottom.x * frame_w), int(p_bottom.y * frame_h))
    cv2.circle(frame, p_top_coord, 3, (0, 255, 255), -1)
    cv2.circle(frame, p_bottom_coord, 3, (0, 255, 255), -1)
    
    eye_height = abs(p_top.y - p_bottom.y)

    if eye_height < EYE_AR_THRESH:
        cv2.putText(frame, "BLINK", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        mouse_x, mouse_y = pyautogui.position()
        cam_x = int((mouse_x / screen_w) * frame_w)
        cam_y = int((mouse_y / screen_h) * frame_h)

        if is_point_in_button(cam_x, cam_y, SWITCH_BUTTON_X, BUTTON_Y): return "switch"
        elif is_point_in_button(cam_x, cam_y, COLOR_BUTTON_X, BUTTON_Y): return "color"
        elif is_point_in_button(cam_x, cam_y, EXIT_BUTTON_X, BUTTON_Y): return "exit"
        elif is_point_in_button(cam_x, cam_y, CLEAR_BUTTON_X, BUTTON_Y): return "clear" # Clear Handle
        else:
            pyautogui.click(duration=0)
            return "click"
    return None

def handle_hand_gestures(frame, hand_results):
    """Handles all hand gesture logic: drawing, erasing, and button interaction."""
    global canvas, xp, yp
    
    if not hand_results.multi_hand_landmarks:
        xp, yp = 0, 0
        return None

    hand_landmarks = hand_results.multi_hand_landmarks[0]
    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    lm_list = []
    for lm_id, lm in enumerate(hand_landmarks.landmark):
        h, w, _ = frame.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        lm_list.append([lm_id, cx, cy])

    if not lm_list: return None

    x_index, y_index = lm_list[INDEX_FINGER_TIP][1:]
    x_thumb, y_thumb = lm_list[THUMB_TIP][1:]

    fingers = []
    tip_ids = [4, 8, 12, 16, 20]
    if lm_list[tip_ids[0]][1] < lm_list[tip_ids[0] - 1][1]: fingers.append(1)
    else: fingers.append(0)
    for i in range(1, 5):
        if lm_list[tip_ids[i]][2] < lm_list[tip_ids[i] - 2][2]: fingers.append(1)
        else: fingers.append(0)
    total_fingers = fingers.count(1)

    click_distance = math.sqrt((x_index - x_thumb)**2 + (y_index - y_thumb)**2)
    if click_distance < 40:
        cv2.circle(frame, (x_index, y_index), 20, (0, 0, 255), cv2.FILLED)
        if is_point_in_button(x_index, y_index, SWITCH_BUTTON_X, BUTTON_Y): return "switch"
        if is_point_in_button(x_index, y_index, COLOR_BUTTON_X, BUTTON_Y): return "color"
        if is_point_in_button(x_index, y_index, EXIT_BUTTON_X, BUTTON_Y): return "exit"
        if is_point_in_button(x_index, y_index, CLEAR_BUTTON_X, BUTTON_Y): return "clear" # Handle Clear

    elif total_fingers == 1 and fingers[1] == 1:
        color = get_current_draw_color()
        cv2.circle(frame, (x_index, y_index), BRUSH_THICKNESS, color, cv2.FILLED)
        if xp == 0 and yp == 0: xp, yp = x_index, y_index
        cv2.line(canvas, (xp, yp), (x_index, y_index), color, BRUSH_THICKNESS)
        xp, yp = x_index, y_index
    elif total_fingers == 5:
        cv2.circle(frame, (x_index, y_index), ERASER_THICKNESS, (255, 255, 255), 2)
        if xp == 0 and yp == 0: xp, yp = x_index, y_index
        cv2.line(canvas, (xp, yp), (x_index, y_index), ERASER_COLOR, ERASER_THICKNESS)
        xp, yp = x_index, y_index
    else:
        xp, yp = 0, 0
    return None

# Loop to run the application
cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN) # full screen er jonno 
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    success, frame = cam.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    action = None
    can_act = (time.time() - last_action_time) > ACTION_COOLDOWN

    cursor_x, cursor_y = (0, 0)
    if current_mode == EYE_TRACKER_MODE:
        mouse_x, mouse_y = pyautogui.position()
        cursor_x = int((mouse_x / screen_w) * frame.shape[1])
        cursor_y = int((mouse_y / screen_h) * frame.shape[0])
    else:
        hand_results = hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            index_tip = hand_results.multi_hand_landmarks[0].landmark[INDEX_FINGER_TIP]
            cursor_x = int(index_tip.x * frame.shape[1])
            cursor_y = int(index_tip.y * frame.shape[0])

    is_over_switch = is_point_in_button(cursor_x, cursor_y, SWITCH_BUTTON_X, BUTTON_Y)
    is_over_color = is_point_in_button(cursor_x, cursor_y, COLOR_BUTTON_X, BUTTON_Y)
    is_over_exit = is_point_in_button(cursor_x, cursor_y, EXIT_BUTTON_X, BUTTON_Y)
    is_over_clear = is_point_in_button(cursor_x, cursor_y, CLEAR_BUTTON_X, BUTTON_Y) # Check hover on Clear

    draw_round_button(frame, SWITCH_BUTTON_X, BUTTON_Y, "SWITCH", highlight=is_over_switch)
    draw_round_button(frame, COLOR_BUTTON_X, BUTTON_Y, "COLOR", highlight=is_over_color)
    draw_round_button(frame, EXIT_BUTTON_X, BUTTON_Y, "EXIT", highlight=is_over_exit)
    draw_round_button(frame, CLEAR_BUTTON_X, BUTTON_Y, "CLEAR", highlight=is_over_clear) # Draw Clear button

    if current_mode == EYE_TRACKER_MODE: # eye tracker mode
        face_results = face_mesh.process(rgb_frame)
        if can_act:
            action = handle_eye_tracking(frame, face_results.multi_face_landmarks)
        frame = cv2.add(frame, canvas)
    else: # hand gesture mode
        if can_act:
            action = handle_hand_gestures(frame, hand_results)
        img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, img_inv)
        frame = cv2.bitwise_or(frame, canvas)
        color_name_text = f"Color: {COLOR_NAMES[current_color_index]}"
        cv2.putText(frame, color_name_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if action:
        last_action_time = time.time()
        if action == "switch":
            current_mode = 1 - current_mode
            mode_message = "Gesture Marker" if current_mode == HAND_GESTURE_MODE else "Eye Tracker"
        elif action == "color":
            color_name = change_color()
            mode_message = f"Color: {color_name}"
        elif action == "exit":
            mode_message = "Exiting..."
            break
        elif action == "clear": # clear kore dibe 
            canvas = CANVAS_BG.copy()
            mode_message = "Canvas Cleared"

    cv2.putText(frame, f"MODE: {mode_message}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    hint_text = "Blink to Click" if current_mode == EYE_TRACKER_MODE else "Thumb+Index to Click | 1 Finger to Draw"
    cv2.putText(frame, hint_text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cam.release()
cv2.destroyAllWindows()
print("Application closed successfully!")