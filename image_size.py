from PIL import Image

# Path to your image file
image_path = "raw_frames/tufts_v_brandeis_frame_990.jpg"

# Open the image
with Image.open(image_path) as img:
    # Get dimensions
    width, height = img.size

print(f"Width: {width}, Height: {height}")

