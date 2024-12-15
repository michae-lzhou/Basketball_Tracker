import cv2
from ultralytics import YOLO
from tqdm import tqdm  # Import tqdm for the progress bar

# Input parameters
input_video_path = "_videos/basketball_game.mp4"  # Path to the saved MP4 file
custom_model_path = "dataset/checkpoints/small-Model2/weights/best.pt"
pre_trained_model_path = "yolov5s.pt"
output_txt_path = "average_coordinates.txt"  # Path to save the average coordinates

# Load YOLO model and set confidence threshold
custom_model = YOLO(custom_model_path)
pre_trained_model = YOLO(pre_trained_model_path)
threshold = 0.25

# Initialize video capture
cap = cv2.VideoCapture(input_video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Cannot open the video file.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames

# Initialize VideoWriter to save output video
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Initialize tqdm progress bar
# progress_bar = tqdm(total=total_frames, desc="Processing Video", unit="frame")

frame_skip = 1
frame_count = 0

# Variables to store cumulative coordinates and weights
total_x, total_y, weight_sum = 0, 0, 0

# Open the text file to write average coordinates
with open(output_txt_path, "w") as file:
    file.write("Frame, Avg_X, Avg_Y\n")  # Write header

    # Process the video frame by frame
    while True:
        ret, frame = cap.read()

        if not ret:
            break  # Break when no more frames to read

        frame_count += 1
        if frame_count % frame_skip == 0:
            # Run YOLO models on the current frame
            custom_results = custom_model(frame)[0]
            pre_trained_results = pre_trained_model(frame)[0]

            # Process custom model results (basketball detection)
            for result in custom_results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                if score > threshold:
                    # Calculate the center of the bounding box
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2
                    # Add weighted coordinates to the cumulative sum
                    total_x += x_center * 2  # Heavier weight for basketball
                    total_y += y_center * 2
                    weight_sum += 2

            # Process pre-trained model results (person detection)
            for result in pre_trained_results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                if score > threshold and pre_trained_results.names[class_id] == "person":
                    # Calculate the center of the bounding box
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2
                    # Add weighted coordinates to the cumulative sum
                    total_x += x_center  # Regular weight for person
                    total_y += y_center
                    weight_sum += 1

            # Compute average coordinates if any detections were made
            if weight_sum > 0:
                avg_x = total_x / weight_sum
                avg_y = total_y / weight_sum
                # Print average coordinates for the current frame
                # print(f"Frame {frame_count}: Avg Coordinates - X: {avg_x}, Y: {avg_y}")
                # Save the average coordinates to the text file
                file.write(f"{frame_count}, {avg_x}, {avg_y}\n")
            else:
                # If no detections, write default values
                # print(f"Frame {frame_count}: No detections.")
                file.write(f"{frame_count}, No detections, No detections\n")

            # Update the progress bar
            # progress_bar.update(1)

# Release resources and close progress bar
cap.release()
# out.release()
# progress_bar.close()
cv2.destroyAllWindows()

print(f"Average coordinates saved to: {output_txt_path}")

