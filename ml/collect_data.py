import cv2
import mediapipe as mp
import csv
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Setup MediaPipe Hands model for detecting a single hand
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# File to save our dataset metrics
DATA_FILE = "sign_data.csv"

def process_landmarks(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    base_x, base_y, base_z = wrist.x, wrist.y, wrist.z

    max_dist = 0.0
    for lm in hand_landmarks.landmark:
        dist = ((lm.x - base_x)**2 + (lm.y - base_y)**2 + (lm.z - base_z)**2)**0.5
        if dist > max_dist:
            max_dist = dist
            
    if max_dist == 0:
        max_dist = 1.0

    features = []
    for lm in hand_landmarks.landmark:
        norm_x = (lm.x - base_x) / max_dist
        norm_y = (lm.y - base_y) / max_dist
        norm_z = (lm.z - base_z) / max_dist
        features.extend([norm_x, norm_y, norm_z])
        
    return features

def collect_data():
    # Prompt the user to enter the sign label
    print("Welcome to Sign Language Data Collector!")
    label = input("Enter the label for the sign you are about to record (e.g., 'A', 'B', 'Hello'): ")

    # Open the standard video camera (0 is usually the built-in webcam)
    cap = cv2.VideoCapture(0)

    print(f"\nRecording data for label: '{label}'")
    print("Instructions:")
    print("- Show the sign to the camera.")
    print("- Press 's' to SAVE the current hand frame as a training sample.")
    print("- Press 'q' to QUIT recording.")

    # Initialize the CSV file with headers if it doesn't already exist
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            # 21 landmarks * 3 coordinates (x, y, z) = 63 features + 1 label (64 columns)
            headers = ["label"] + [f"feature_{i}" for i in range(63)]
            writer.writerow(headers)

    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab camera frame")
            break
            
        # Flip the frame horizontally for selfie-view validation
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image (OpenCV default) to RGB (MediaPipe requirement)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and try to detect hands
        results = hands.process(rgb_frame)
        
        # Optional: Draw the hand landmarks on the frame to give user visual feedback
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS
                )
                
        # Display the frame and data count
        cv2.putText(frame, f"Samples for '{label}': {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Sign Language Data Collector', frame)
        
        # Check keystrokes
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            # Press 's' to save the current frame's landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Normalize and flatten the 21 landmarks into a single array
                    features = process_landmarks(hand_landmarks)
                    row = [label] + features  # First column is our target label
                    
                    # Append it to the CSV
                    with open(DATA_FILE, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(row)
                    
                    count += 1
                    print(f"Saved sample #{count} for '{label}'")
            else:
                print("No hand detected! Try again.")
                
        elif key == ord('q'):
            # Press 'q' to quit
            break

    # Clean up the camera resources
    cap.release()
    cv2.destroyAllWindows()
    print("Data collection stopped.")

if __name__ == "__main__":
    collect_data()
