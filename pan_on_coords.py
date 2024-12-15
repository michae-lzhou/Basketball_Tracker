import cv2
import numpy as np

# Load the video
cap = cv2.VideoCapture("_videos/basketball_game.mp4")

# Get frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Crop width: half of the frame width (adjustable based on your zoom factor)
crop_width = frame_width // 2

# Calculate crop height based on the aspect ratio
crop_height = crop_width / (frame_width / frame_height)

# Preset crop positions (left, middle, right)
left_crop_start = 0
middle_crop_start = (frame_width - crop_width) // 2
right_crop_start = frame_width - crop_width

# Set the initial position to middle
current_crop_start = middle_crop_start

# Set the initial target to the middle crop position
target_crop_start = middle_crop_start

# Transition parameters
max_transition_speed = 10  # The maximum number of pixels to move per frame
min_transition_speed = 1   # The minimum number of pixels to move per frame
transition_factor = 0.05   # Factor to adjust speed based on distance

# Initialize vertical offset (starts at 0)
vertical_offset = 0

# Read the average coordinates from the text file
with open("average_coordinates.txt", "r") as file:
    avg_coords = file.readlines()

# Loop through each frame
frame_num = 0
avg_coord_index = 1  # Index to keep track of the average coordinates per frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # If there are no more coordinates, break the loop
    if avg_coord_index >= len(avg_coords):
        break

    # Extract the average X and Y coordinates for this frame
    frame_num_txt, x_coord, y_coord = avg_coords[avg_coord_index].strip().split(',')

    x_coord = float(x_coord)
    y_coord = float(y_coord)

    # Determine pan position based on the X-coordinate
    if x_coord < frame_width * 5 / 11:
        # Pan to the left
        target_crop_start = left_crop_start
    elif x_coord > frame_width * 6 / 11:
        # Pan to the right
        target_crop_start = right_crop_start
    else:
        # Pan to the middle
        target_crop_start = middle_crop_start

    # Compute the crop window based on the current position
    crop_top = int((frame_height - crop_height) // 2) + vertical_offset
    crop_bottom = int((frame_height - crop_height) // 2) + vertical_offset + int(crop_height)

    # Ensure the vertical crop stays within bounds
    if crop_top < 0:
        crop_top = 0
        crop_bottom = int(crop_height)
    if crop_bottom > frame_height:
        crop_bottom = frame_height
        crop_top = frame_height - int(crop_height)

    # Crop the frame to the desired dimensions (crop_width and crop_height)
    cropped_frame = frame[crop_top:crop_bottom, current_crop_start:current_crop_start + crop_width]

    # Show the cropped frame
    cv2.imshow("Pan Effect", cropped_frame)

    # Wait for the next frame and capture key press
    key = cv2.waitKey(1) & 0xFF

    # Calculate the distance between the current and target position
    distance = abs(current_crop_start - target_crop_start)

    # Adjust the transition speed based on the distance
    transition_speed = int(min(max_transition_speed, distance * transition_factor))

    # Smooth transition to the target crop position
    if current_crop_start != target_crop_start:
        if current_crop_start < target_crop_start:
            current_crop_start += transition_speed
            if current_crop_start > target_crop_start:
                current_crop_start = target_crop_start
        elif current_crop_start > target_crop_start:
            current_crop_start -= transition_speed
            if current_crop_start < target_crop_start:
                current_crop_start = target_crop_start

    # Increment the frame number and move to the next set of coordinates
    frame_num += 1
    avg_coord_index += 1

# Release the video capture and close the OpenCV window
cap.release()
cv2.destroyAllWindows()

