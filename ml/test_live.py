import cv2
import mediapipe as mp
import pickle
import numpy as np
import collections
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

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


# Load the trained model from ml/model/sign_model.pkl
# Using relative path assuming we run this script from the ml directory
MODEL_PATH = "model/sign_model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except FileNotFoundError:
    print(f"Error: Model not found at {MODEL_PATH}.")
    print("Please ensure you have trained the model and it is saved correctly.")
    exit()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Setup MediaPipe Hands model for detecting a single hand
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Open the webcam
cap = cv2.VideoCapture(0)

# Buffer for prediction smoothing
prediction_buffer = collections.deque(maxlen=5)
stable_prediction = "Waiting..."

print("Starting real-time sign language detection...")
print("Press 'q' to QUIT.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab camera frame")
        break
        
    # Flip the frame horizontally for selfie-view validation
    frame = cv2.flip(frame, 1)
    
    # Convert the BGR image to RGB (MediaPipe requirement)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and try to detect hands
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )
            
            # Extract 21 hand landmarks and flatten them
            landmarks_flattened = process_landmarks(hand_landmarks)
            
            # Convert list to a numpy array and reshape for prediction (1 sample, n features)
            landmarks_array = np.array(landmarks_flattened).reshape(1, -1)
            
            try:
                # To avoid scikit-learn warnings about feature names
                # not important for testing locally but good practice
                
                # Run prediction using the trained model
                prediction = model.predict(landmarks_array)
                predicted_label = str(prediction[0])
                
                # Try to get prediction probability if the model supports it
                confidence = 0.0
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(landmarks_array)
                    confidence = float(np.max(proba[0]))
                
                prediction_buffer.append((predicted_label, confidence))
                
                labels = [p[0] for p in prediction_buffer]
                most_common = max(set(labels), key=labels.count)
                
                # Only update stable prediction if we have a consensus and confidence is decent
                # Alternatively, you can average the confidence
                avg_confidence = np.mean([p[1] for p in prediction_buffer if p[0] == most_common])
                
                if labels.count(most_common) >= 3 and avg_confidence > 0.6:
                    stable_prediction = f"{most_common} ({avg_confidence:.2f})"
                else:
                    stable_prediction = f"Thinking... ({most_common})"
                    
                # Display predicted label on screen
                cv2.putText(
                    frame, 
                    f"Sign: {stable_prediction}", 
                    (10, 50),                 # Position (x, y)
                    cv2.FONT_HERSHEY_SIMPLEX, # Font style
                    1.0,                      # Font scale
                    (0, 255, 0),              # Green color
                    2,                        # Thickness
                    cv2.LINE_AA               # Anti-aliasing
                )
            except Exception as e:
                # Handle potential prediction errors
                cv2.putText(
                    frame,
                    "Prediction Error",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
                
    else:
        # Clear buffer if no hand is detected to reset smoothing
        prediction_buffer.clear()
        stable_prediction = "Waiting..."
        # Handle case when no hand is detected
        cv2.putText(
            frame, 
            "No hand detected", 
            (10, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 0, 255),  # Red color
            2
        )

    # Show webcam window in real time
    cv2.imshow('Sign Language Live Test', frame)
    
    # Check keystrokes, press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()
print("Live test stopped.")
