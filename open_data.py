import os
import random
import matplotlib.pyplot as plt
import cv2


# Display 8 random images


label_names=["Basketball"]


def get_annoations(original_img,label_file):


    with open(label_file, 'r') as file:
            lines = file.readlines()
        
    annotations = []
    
    for line in lines:
        values = line.split()
        label = values[0]
        x, y, w, h = map(float, values[1:])
        annotations.append((label, x, y, w, h))


    return annotations


def put_annoations_in_image(image,annotations):
    
    H, W, _ = image.shape
    for annotation in annotations:
            label, x, y, w, h = annotation
            print(label, x, y, w, h)
            label_name = label_names[int(label)]
            
            #Convert YOLO coordinates to pixel coordinates
            x1 = int((x - w / 2) * W)
            y1 = int((y - h / 2) * H)
            x2 = int((x + w / 2) * W)
            y2 = int((y + h / 2) * H)
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (200, 200, 0), 1)
            
            # Display label
            cv2.putText(image, label_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 2), cv2.LINE_AA


    return image




def display_random_images(folder_path, num_images, label_folder):
    # Get list of all image filenames in the folder
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    # Randomly select num_images filenames
    selected_images = random.sample(image_files, num_images)


    # Create a subplot grid
    #fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # Adjusted to display 8 images in a 2x4 grid
    #fig.suptitle('Randomly Selected Images')


    # Iterate through the selected images and display them
    for i, image_file in enumerate(selected_images):
        #row = i // 4
        #col = i % 4
        
        img = cv2.imread(os.path.join(folder_path, image_file))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB (matplotlib uses RGB)
        
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_file_path = os.path.join(label_folder, label_file)
        
        # read annotations
        annotations_Yolo_format = get_annoations(img,label_file_path)
        #print(annotations_Yolo_format)
        


        # put bounding boxes
        image_with_anotations = put_annoations_in_image(img,annotations_Yolo_format)
        print(image_with_anotations.shape)
        cv2.imshow("img no. " + str(i),image_with_anotations)
        cv2.waitKey(0)
        
        
        #axes[row, col].imshow(image_with_anotations)
        #axes[row, col].axis('off')


    plt.show()


# Replace 'folder_path' with the path to your folder containing images
images_path = 'dataset/train/images'
label_folder = 'dataset/train/labels'


num_images = 4  # Adjusted to display 8 images
display_random_images(images_path, num_images, label_folder)

