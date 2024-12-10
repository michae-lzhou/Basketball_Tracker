import os
import random
import cv2
from tqdm import tqdm

label_names = ["Basketball"]

def draw_bounding_box(event, x, y, flags, param):
    """Mouse callback function to draw bounding boxes."""
    global drawing, ix, iy, ex, ey, drawn_box

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            ex, ey = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        ex, ey = x, y
        drawn_box = (ix, iy, ex, ey)

def get_annotations(label_file):
    """Read YOLO annotations from a label file."""
    with open(label_file, 'r') as file:
        lines = file.readlines()

    annotations = []
    for line in lines:
        values = line.split()
        label = values[0]
        x, y, w, h = map(float, values[1:])
        annotations.append((label, x, y, w, h))

    return annotations

def process_random_images(image_folder, label_folder, num_images=10):
    """Process 10 random images with their labels."""
    global drawing, ix, iy, ex, ey, drawn_box
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    random_images = random.sample(image_files, min(num_images, len(image_files)))

    print(f"Processing {len(random_images)} random images...")
    for image_file in random_images:
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_file_path = os.path.join(label_folder, label_file)

        if not os.path.exists(label_file_path):
            print(f"No label file found for {image_file}. Skipping.")
            continue

        # Load image
        img_path = os.path.join(image_folder, image_file)
        img = cv2.imread(img_path)
        clone = img.copy()

        annotations = get_annotations(label_file_path)

        # Draw existing annotations
        H, W, _ = img.shape
        for annotation in annotations:
            label, x, y, w, h = annotation
            x1 = int((x - w / 2) * W)
            y1 = int((y - h / 2) * H)
            x2 = int((x + w / 2) * W)
            y2 = int((y + h / 2) * H)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.namedWindow("Draw Bounding Box")
        cv2.setMouseCallback("Draw Bounding Box", draw_bounding_box)

        no_basketball = False

        while True:
            temp_img = img.copy()
            if drawn_box is not None:
                cv2.rectangle(temp_img, (ix, iy), (ex, ey), (255, 0, 0), 2)
            cv2.imshow("Draw Bounding Box", temp_img)

            key = cv2.waitKey(1)
            if key == ord('q'):  # Quit
                cv2.destroyAllWindows()
                return
            elif key == ord('s'):  # Save the manually drawn bounding box
                if drawn_box is not None:
                    x1, y1, x2, y2 = drawn_box
                    cx = ((x1 + x2) / 2) / W
                    cy = ((y1 + y2) / 2) / H
                    w = abs(x2 - x1) / W
                    h = abs(y2 - y1) / H
                    label_str = f"0 {cx} {cy} {w} {h}\n"
                    with open(label_file_path, 'w') as f:
                        f.write(label_str)
                    print(f"Saved manually drawn bounding box for {label_file}.")
                break
            elif key == ord('c'):  # Confirm existing bounding boxes
                print(f"Bounding boxes for {label_file} confirmed as correct.")
                break
            elif key == ord('n'):  # Indicate no basketball is present
                with open(label_file_path, 'w') as f:
                    f.write("")
                print(f"No basketball indicated for {label_file}.")
                no_basketball = True
                break

        if no_basketball:
            continue

        cv2.destroyAllWindows()

# Global variables for mouse callback
drawing = False
ix, iy, ex, ey = -1, -1, -1, -1
drawn_box = None

# Replace with the path to your dataset
images_path = 'dataset/train/images'
labels_path = 'dataset/train/labels'

process_random_images(images_path, labels_path)
