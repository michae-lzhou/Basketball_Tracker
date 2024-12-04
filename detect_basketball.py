# NOTE: DO NOT RUN THIS FILE, THE IMAGE DETECTION IS COMPLETELY OFF

exit()

import cv2
import os
import numpy as np

# Create the testing_data folder if it doesn't exist
output_folder = 'testing_data'
os.makedirs(output_folder, exist_ok=True)

# Define the color range for basketball detection in HSV space
lower_orange = np.array([5, 50, 50])  # Lower bound for orange color
upper_orange = np.array([15, 255, 255])  # Upper bound for orange color

# Function to detect basketball in a frame
def detect_ball(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the basketball color
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Optional: Apply morphological operations to clean the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ball_position = None
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Minimum contour area threshold
            # Get bounding box for the detected ball
            x, y, w, h = cv2.boundingRect(contour)
            ball_position = (x, y, w, h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame, ball_position

# Load and process each frame in the raw_frames folder
raw_frames_folder = 'raw_frames'

frame_files = os.listdir(raw_frames_folder)
for frame_file in frame_files:
    if frame_file.endswith('.jpg') or frame_file.endswith('.png'):  # Only process image files
        frame_path = os.path.join(raw_frames_folder, frame_file)
        
        # Load the frame
        frame = cv2.imread(frame_path)

        # Detect the ball in the frame
        detected_frame, ball_position = detect_ball(frame)

        # If a basketball was detected, save the frame to the testing_data folder
        if ball_position is not None:
            detected_frame_path = os.path.join(output_folder, f'detected_{frame_file}')
            cv2.imwrite(detected_frame_path, detected_frame)
            print(f"Saved detected frame: {detected_frame_path}")
        else:
            print(f"No ball detected!")

print("Detection complete!")
