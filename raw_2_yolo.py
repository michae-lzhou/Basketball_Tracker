import os

# Define image dimensions
IMAGE_WIDTH = 1920  # Replace with your actual image width
IMAGE_HEIGHT = 1080  # Replace with your actual image height

# Input file containing your original annotations
input_file = "bbox_txt/tufts_v_brandeis_bounding_boxes.txt"
# Output folder for YOLO annotations
output_folder = "yolo_labels"
os.makedirs(output_folder, exist_ok=True)

# Process each line in the input file
with open(input_file, "r") as file:
    for line in file:
        # Parse the input line
        parts = line.strip().split(":")
        image_file = parts[0].strip()
        bbox_data = parts[1].strip()

        # Extract x1, y1, width, height
        x1, y1, width, height = map(int, bbox_data.split(","))
        
        # Calculate YOLO format (normalized values)
        x_center = (x1 + width / 2) / IMAGE_WIDTH
        y_center = (y1 + height / 2) / IMAGE_HEIGHT
        norm_width = width / IMAGE_WIDTH
        norm_height = height / IMAGE_HEIGHT

        # Create YOLO formatted string (class_id = 0 for basketballs)
        yolo_line = f"0 {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n"

        # Save the YOLO annotation
        label_file = os.path.join(output_folder, os.path.splitext(image_file)[0] + ".txt")
        with open(label_file, "w") as label_out:
            label_out.write(yolo_line)

print(f"YOLO labels saved to {output_folder}")

