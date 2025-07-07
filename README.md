# eye_and_hand_gesture_project

AI Virtual Control 👁️🤚✨
Welcome to my AI Virtual Control project! This is a Python application I built that transforms your webcam into a futuristic controller, allowing you to manipulate your computer's cursor and draw on a virtual canvas using just your eye movements and hand gestures.


🚀 Features
I've packed this application with several cool features:

Dual Control Modes: Seamlessly switch between two powerful control methods:

Eye Tracking: Control the cursor with the movement of your iris.

Hand Gestures: Use your hand as a virtual mouse and drawing tool.

Full Cursor Control: The script gives you full control over your system's mouse cursor in real-time.

Intuitive Clicking:

In Eye Mode, simply blink to perform a click.

In Hand Mode, pinch your thumb and index finger together to click.

Virtual Drawing Canvas: Express your creativity on a full-screen digital canvas.

Dynamic Color Palette: Cycle through a variety of colors for drawing, including a special ✨Rainbow Mode✨ that shifts colors dynamically!

Eraser Functionality: Made a mistake? Just raise all five fingers to activate the eraser and clear parts of your drawing.

Interactive UI: On-screen buttons allow you to easily switch modes, change colors, clear the entire canvas, or exit the application.

🛠️ How It Works
I built this system by combining several powerful computer vision libraries. Here’s a breakdown of how each mode functions:

👁️ Eye Tracking Mode
Face Mesh Detection: I use the MediaPipe Face Mesh model to detect a detailed map of 478 landmarks on the face in real-time. I specifically enabled refine_landmarks=True to get access to the high-precision iris coordinates.

Iris-to-Cursor Mapping: I isolate the coordinates of the iris. These coordinates (which are normalized between 0 and 1) are then scaled to match my computer screen's full resolution.

Movement Amplification & Smoothing: To make the control feel more natural and less jittery, I implemented two key things:

Amplification: I magnify the detected iris movements so that small, comfortable eye motions result in significant cursor travel.

Smoothing: I apply an exponential moving average to the cursor's position. This averages out the current position with previous ones, resulting in a much smoother and more stable pointer.

Blink Detection: To detect a click, I constantly measure the vertical distance between the upper and lower eyelid landmarks. When this distance falls below a specific, tiny threshold (EYE_AR_THRESH), the script registers it as a blink and triggers a click action.

🤚 Hand Gesture Mode
Hand Landmark Detection: I use the MediaPipe Hands model to detect the 21 key landmarks of a single hand.

Gesture Recognition: I wrote custom logic to interpret the position of these landmarks as specific commands:

Clicking (Selection): When the Euclidean distance between the tip of the thumb (landmark 4) and the tip of the index finger (landmark 8) is very small, it's recognized as a pinch gesture, which triggers a click. This is used to press the on-screen buttons.

Drawing: The "drawing" gesture is recognized when only the index finger is extended. In this state, the script draws a line on the virtual canvas, following the path of the index fingertip.

Erasing: The "erasing" gesture is recognized when all five fingers are extended. This activates a larger, eraser-tipped brush that draws in the background color.
Idle: Any other hand pose is considered idle, allowing the user to move their hand without drawing or clicking.

💻 Technology Stack
This project was built entirely in Python, relying on these amazing open-source libraries:
OpenCV (cv2): For capturing webcam feed, handling image processing, drawing UI elements (buttons, text), and displaying the final output.
MediaPipe: For the heavy lifting of AI-based detection. I used its FaceMesh and Hands solutions.
PyAutoGUI: To programmatically control the system's mouse (moving the cursor and performing clicks).
NumPy: For creating and managing the image arrays, especially the black background for the drawing canvas.

🔧 Setup and Installation
To run this project on your own machine, follow these steps:

1. Prerequisites
Python 3.7 or newer.
A webcam connected to your computer.

2. Clone the Repository
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME

3. Install Dependencies
I've listed all the required packages in the requirements.txt file. You can install them all with a single command:

pip install -r requirements.txt

(Note: If you don't have a requirements.txt file, you can create one or install the packages manually: pip install opencv-python mediapipe pyautogui numpy)

▶️ How to Run
Once the setup is complete, just run the following command in your terminal:
python your_script_name.py

The application will launch in full-screen mode. Enjoy!

🎮 Controls Summary
Global: Press the 'q' key at any time to quit the application.
Buttons: Use the on-screen buttons to SWITCH modes, change COLOR, CLEAR the canvas, or EXIT.

In Eye Tracking Mode:
Move Cursor: Look around the screen.
Click: Perform a quick blink.

In Hand Gesture Mode:
Move Cursor: Move your hand around.
Click (Select): Pinch your thumb and index finger together.
Draw: Raise only your index finger.
Erase: Raise all five fingers.
