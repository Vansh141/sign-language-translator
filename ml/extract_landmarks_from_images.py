import cv2
import mediapipe as mp
import csv
import os
import sys

# ==========================================
# CONFIGURATION
# Modify these paths as needed.
# ==========================================
# The main directory containing subfolders for each sign (e.g., A/, B/, C/)
DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset")
# The CSV file to append the extracted landmarks to
OUTPUT_CSV = "sign_data.csv"
# Print a progress update every N images
LOG_INTERVAL = 100 
# ==========================================

def extract_landmarks(dataset_path, output_csv):
    """
    Reads images from a categorized dataset, extracts MediaPipe hand landmarks,
    and appends them to a CSV file for training.
    """
    
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
    
    # 1. Initialize MediaPipe Hands in static image mode
    # Static mode is optimized for processing independent images rather than a video stream
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True, 
        max_num_hands=1, 
        min_detection_confidence=0.5
    )

    # 2. Track our progress
    total_processed = 0
    total_saved = 0
    total_skipped = 0
    
    # Check if the dataset folder exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        print("Please create this folder and organize your images like: dataset/A/image1.jpg")
        sys.exit(1)

    print(f"Starting landmark extraction from '{dataset_path}'...")
    print(f"Appending valid rows to '{output_csv}'...\n")

    # 3. Create the CSV file with headers if it doesn't already exist
    file_exists = os.path.isfile(output_csv)
    if not file_exists:
        with open(output_csv, mode="w", newline="") as f:
            writer = csv.writer(f)
            headers = ["label"] + [f"feature_{i}" for i in range(63)]
            writer.writerow(headers)
            
    # Open the CSV file in 'append' mode ('a')
    with open(output_csv, mode="a", newline="") as f:
        writer = csv.writer(f)

        # Loop through every class folder (A, B, C...) in the dataset
        for label_name in os.listdir(dataset_path):
            class_dir = os.path.join(dataset_path, label_name)
            
            # Skip floating files; only process directories
            if not os.path.isdir(class_dir):
                continue
                
            # Loop through all images inside this specific class folder
            for filename in os.listdir(class_dir):
                filepath = os.path.join(class_dir, filename)
                
                # Simple check to skip hidden files or non-images if necessary
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                total_processed += 1
                
                # 4. Read the image using OpenCV
                image = cv2.imread(filepath)
                if image is None:
                    # Skip corrupt or unreadable images gracefully
                    total_skipped += 1
                    continue
                    
                # MediaPipe requires RGB format, whereas OpenCV loads in BGR
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 5. Process the image to find hands
                results = hands.process(rgb_image)
                
                # 6. Extract landmarks if a hand was found
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Normalize and flatten the landmarks securely
                        features = process_landmarks(hand_landmarks)
                        row = [label_name] + features
                            
                        # Save the row robustly (1 label + 63 features = 64 columns)
                        writer.writerow(row)
                        total_saved += 1
                        
                        # Since we enforce max_num_hands=1, we can break after the first found
                        break
                else:
                    # If MediaPipe couldn't confidently spot a hand in this specific image
                    total_skipped += 1
                
                # 7. Print progress cleanly
                if total_processed % LOG_INTERVAL == 0:
                    print(f"Processed: {total_processed} | Saved: {total_saved} | Skipped: {total_skipped}")

    # 8. Clean up MediaPipe engine resources
    hands.close()

    print("\n" + "="*40)
    print("EXTRACTION COMPLETE!")
    print(f"Total Images Viewed : {total_processed}")
    print(f"Total Rows Appended : {total_saved}")
    print(f"Total Skipped/Failed: {total_skipped}")
    print("="*40)

if __name__ == "__main__":
    extract_landmarks(DATASET_PATH, OUTPUT_CSV)
