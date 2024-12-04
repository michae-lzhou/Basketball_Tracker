import cv2
import os

raw_frames_folder = 'raw_frames'
# Change the video name here for different videos
vid_name = 'tufts_v_brandeis'
output_folder = 'testing_data'
txt_folder = 'bbox_txt'
os.makedirs(output_folder, exist_ok=True)

frame_files = os.listdir(raw_frames_folder)
for frame_file in frame_files:
    if frame_file.endswith('.jpg') and frame_file.startswith(vid_name):
        frame_path = os.path.join(raw_frames_folder, frame_file)
        
        # Load the frame
        frame = cv2.imread(frame_path)
        
        # Manually label the frame using selectROI
        bbox = cv2.selectROI("Select Basketball", frame, fromCenter=False, showCrosshair=True)
        
        if bbox != (0, 0, 0, 0):  # Check if a bounding box was selected
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Save the labeled frame
            labeled_frame_path = os.path.join(output_folder, f'labeled_{frame_file}')
            cv2.imwrite(labeled_frame_path, frame)
            
            # Save coordinates to a text file or CSV
            coordinates_file = os.path.join(txt_folder, f'{vid_name}_bounding_boxes.txt')
            with open(coordinates_file, 'a') as f:
                f.write(f'{frame_file}: {x}, {y}, {w}, {h}\n')
            
            print(f"Saved labeled frame: {labeled_frame_path}")
        
        else:
            print(f"No bounding box selected for {frame_file}")
        
        # Optional: Show the labeled frame for visual feedback
        cv2.imshow("Labeled Frame", frame)
        cv2.waitKey(0)

cv2.destroyAllWindows()
print("Labeling complete!")
