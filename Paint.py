import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Drawing colors (blue, green, red, yellow, purple) and eraser (black)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)]
eraser_color = (0, 0, 0)
current_color = colors[0]
drawing = False
erasing = False
prev_pos = None  # Store the previous position for drawing smooth lines
eraser_thickness = 20  # Default eraser thickness
drawing_thickness = 8  # Default drawing thickness

# Initialize canvas
canvas = np.zeros((480, 640, 3), dtype="uint8")

def get_finger_tip_position(hand_landmarks, idx):
    return int(hand_landmarks.landmark[idx].x * 640), int(hand_landmarks.landmark[idx].y * 480)

def select_color(x, y):
    global current_color, erasing
    # Assuming the top left 5 squares (50x50 each) contain color options
    if y < 50:
        erasing = False  # Stop erasing when selecting a color
        for i in range(5):
            if 50 * i < x < 50 * (i + 1):
                current_color = colors[i]
    # Check if the user selects the eraser (assuming it's in the 6th position)
    if y < 50 and 250 < x < 300:
        erasing = True
        current_color = eraser_color

cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to find hands
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            index_tip = get_finger_tip_position(hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP)
            middle_tip = get_finger_tip_position(hand_landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP)

            # Check if only the index finger is up
            index_finger_up = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < \
                              hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y
            middle_finger_up = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < \
                               hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y

            if index_finger_up and not middle_finger_up:  # Only index finger is up
                if drawing or erasing:  # Continue drawing or erasing
                    if prev_pos:  # Draw or erase from the previous position to the current position
                        thickness = eraser_thickness if erasing else drawing_thickness
                        cv2.line(canvas, prev_pos, index_tip, current_color, thickness)
                    prev_pos = index_tip  # Update the previous position
                else:  # Start drawing or erasing
                    drawing = True
                    prev_pos = index_tip  # Set initial position
            else:
                drawing = False
                prev_pos = None  # Reset previous position

            # Color and eraser selection logic
            if index_finger_up and middle_finger_up:  # Both index and middle fingers are up
                select_color(index_tip[0], index_tip[1])

            # Draw the selected color indicator on the index fingertip
            cv2.circle(frame, index_tip, 10, current_color, -1)

            # Draw the hand landmarks for visualization
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Overlay the canvas on the frame
    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Draw color selection bar and eraser
    for i, color in enumerate(colors):
        cv2.rectangle(frame, (50 * i, 0), (50 * (i + 1), 50), color, -1)
    cv2.rectangle(frame, (250, 0), (300, 50), (200, 200, 200), -1)  # Eraser option (gray rectangle)

    # Show current color and thickness
    cv2.putText(frame, f'Color: {current_color}', (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'Drawing Thickness: {drawing_thickness}px', (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'Eraser Thickness: {eraser_thickness}px', (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Drawing Panel", frame)

    # Adjust thickness based on key press
    key = cv2.waitKey(1)
    if key & 0xFF == ord('+'):  # Increase thickness
        if current_color == eraser_color:  # Increase eraser thickness
            eraser_thickness = min(eraser_thickness + 5, 50)  # Max thickness of 50
        else:  # Increase drawing thickness
            drawing_thickness = min(drawing_thickness + 5, 50)  # Max thickness of 50
    elif key & 0xFF == ord('-'):  # Decrease thickness
        if current_color == eraser_color:  # Decrease eraser thickness
            eraser_thickness = max(eraser_thickness - 5, 5)  # Min thickness of 5
        else:  # Decrease drawing thickness
            drawing_thickness = max(drawing_thickness - 5, 5)  # Min thickness of 5
    elif key & 0xFF == 27:  # Press 'Esc' to exit
        break

    # Close the program if the 'X' button on the window is clicked
    if cv2.getWindowProperty("Drawing Panel", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
