import cv2
import mediapipe as mp
import numpy as np
import time
import math

# Initialization
# UI & Control Constants
# Layout - buttons and their positions
BUTTON_RADIUS = 60
BUTTON_Y = 80
COLOR_BUTTON_X = 570
CLEAR_BUTTON_X = 710
CLICK_RADIUS = 70

# Colors
BUTTON_COLOR = (100, 100, 100)
BUTTON_TEXT_COLOR = (255, 255, 255)
HIGHLIGHT_COLOR = (0, 255, 0)
CANVAS_BG = np.zeros((720, 1280, 3), np.uint8)
COLOR_PALETTE = [
    (0, 255, 0),      # Green
    (255, 0, 0),      # Blue
    (0, 0, 255),      # Red
    (255, 255, 0),    # Cyan
    (42, 42, 165),    # Brown
    (255, 0, 255),    # Magenta
    (128, 0, 128),    # Purple
    (128, 128, 128),  # Grey
    (0, 0, 0)         # Rainbow (Special case)
]
COLOR_NAMES = ["Green", "Blue", "Red", "Sky Blue", "Brown", "Pink", "Purple", "Grey", "Rainbow ✨"]
ERASER_COLOR = (0, 0, 0)
BRUSH_THICKNESS = 15
ERASER_THICKNESS = 50

current_color_index = 0
DRAW_COLOR = COLOR_PALETTE[current_color_index]
rainbow_hue = 0
xp, yp = 0, 0
status_message = "Gesture Marker Enabled"
ACTION_COOLDOWN = 0.5
last_action_time = 0

# camrera input 
cam = cv2.VideoCapture(0)
cam.set(3, 1280)
cam.set(4, 720)

# Hands detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Landmark constants for clarity
INDEX_FINGER_TIP = 8
THUMB_TIP = 4

# Canvas
canvas = CANVAS_BG.copy()
WINDOW_NAME = 'AI Virtual Hand Marker'

def draw_round_button(frame, center_x, center_y, text, highlight=False):
    """Draws a round button, highlighting it if active."""
    overlay = frame.copy()
    
    button_color = HIGHLIGHT_COLOR if highlight else BUTTON_COLOR
    cv2.circle(overlay, (center_x, center_y), BUTTON_RADIUS, button_color, -1)
    
    alpha = 0.6  # Transparency amount
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # border and text
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

def handle_hand_gestures(frame, hand_results):
    """Handles all hand gesture logic: drawing, erasing, and button interaction."""
    global canvas, xp, yp

    if not hand_results.multi_hand_landmarks:
        xp, yp = 0, 0
        return None, None 

    hand_landmarks = hand_results.multi_hand_landmarks[0]
    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    lm_list = []
    for lm_id, lm in enumerate(hand_landmarks.landmark):
        h, w, _ = frame.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        lm_list.append([lm_id, cx, cy])

    if not lm_list: return None, None

    x_index, y_index = lm_list[INDEX_FINGER_TIP][1:]
    x_thumb, y_thumb = lm_list[THUMB_TIP][1:]

    # highlighting buttons on hover
    cursor_pos = (x_index, y_index)

    # counting logic
    fingers = []
    tip_ids = [4, 8, 12, 16, 20]
    # Thumb 
    if lm_list[tip_ids[0]][1] < lm_list[tip_ids[0] - 1][1]:
        fingers.append(1)
    else:
        fingers.append(0)
    # Other 4 fingers (based on y-position)
    for i in range(1, 5):
        if lm_list[tip_ids[i]][2] < lm_list[tip_ids[i] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)
    total_fingers = fingers.count(1)

    # click gesture
    click_distance = math.sqrt((x_index - x_thumb)**2 + (y_index - y_thumb)**2)
    if click_distance < 40:
        cv2.circle(frame, (x_index, y_index), 20, (0, 0, 255), cv2.FILLED)
        if is_point_in_button(x_index, y_index, COLOR_BUTTON_X, BUTTON_Y): return "color", cursor_pos
        if is_point_in_button(x_index, y_index, CLEAR_BUTTON_X, BUTTON_Y): return "clear", cursor_pos

    # drawwing logic
    elif total_fingers == 1 and fingers[1] == 1:
        color = get_current_draw_color()
        cv2.circle(frame, (x_index, y_index), BRUSH_THICKNESS, color, cv2.FILLED)
        if xp == 0 and yp == 0: xp, yp = x_index, y_index
        cv2.line(canvas, (xp, yp), (x_index, y_index), color, BRUSH_THICKNESS)
        xp, yp = x_index, y_index
        
    # erasing logic
    elif total_fingers == 5:
        overlay = frame.copy()
        cv2.circle(overlay, (x_index, y_index), ERASER_THICKNESS, (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        if xp == 0 and yp == 0: xp, yp = x_index, y_index
        cv2.line(canvas, (xp, yp), (x_index, y_index), ERASER_COLOR, ERASER_THICKNESS)
        xp, yp = x_index, y_index
        
    # Reset drawing 
    else:
        xp, yp = 0, 0

    return None, cursor_pos
# main loop
cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    success, frame = cam.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    action = None
    can_act = (time.time() - last_action_time) > ACTION_COOLDOWN

    hand_results = hands.process(rgb_frame)
    if can_act:
        action, cursor_pos = handle_hand_gestures(frame, hand_results)
    else:
        _, cursor_pos = handle_hand_gestures(frame, hand_results)

    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, img_inv)
    frame = cv2.bitwise_or(frame, canvas)

    cursor_x, cursor_y = (0, 0)
    if cursor_pos:
        cursor_x, cursor_y = cursor_pos

    # button hover
    is_over_color = is_point_in_button(cursor_x, cursor_y, COLOR_BUTTON_X, BUTTON_Y)
    is_over_clear = is_point_in_button(cursor_x, cursor_y, CLEAR_BUTTON_X, BUTTON_Y)

    # buttons
    draw_round_button(frame, COLOR_BUTTON_X, BUTTON_Y, "COLOR", highlight=is_over_color)
    draw_round_button(frame, CLEAR_BUTTON_X, BUTTON_Y, "CLEAR", highlight=is_over_clear)

    # instructions and guidelines
    cv2.line(frame, (250, 150), (250, 550), (255, 255, 0), 4)
    cv2.putText(frame, "Please Stand Here", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.arrowedLine(frame, (150, 350), (240, 350), (255, 255, 0), 4, tipLength=0.3)

    cv2.putText(frame, "Draw on this side", (frame.shape[1] - 300, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    color_name_text = f"Color: {COLOR_NAMES[current_color_index]}"
    cv2.putText(frame, color_name_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # button press action
    if action:
        last_action_time = time.time()
        if action == "color":
            color_name = change_color()
            status_message = f"Color: {color_name}"
        elif action == "clear":
            canvas = CANVAS_BG.copy()
            status_message = "Canvas Cleared"

    # display status 
    cv2.putText(frame, f"STATUS: {status_message}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    hint_text = "Thumb+Index to Click | 1 Finger to Draw | 5 Fingers to Erase"
    cv2.putText(frame, hint_text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow(WINDOW_NAME, frame)

    # q to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
cam.release()
cv2.destroyAllWindows()
print("Application closed successfully!")