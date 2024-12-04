import cv2
import os

def preprocess():
    # Set up output folder
    output_folder = "testing_data"
    os.makedirs(output_folder, exist_ok=True)

    # Load the video
    vid_name = "tufts_v_brandeis"
    video_path = f"_videos/{vid_name}.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Cannot open video.")
        exit()
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save every nth frame for testing (e.g., every 30th frame)
        if frame_count % 30 == 0:
            frame_filename = os.path.join(output_folder,
                                          f"{vid_name}_frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved {frame_filename}")
        
        frame_count += 1
    
    cap.release()
    print("Preprocessed frames saved!")

if __name__ == "__main__":
    preprocess()
