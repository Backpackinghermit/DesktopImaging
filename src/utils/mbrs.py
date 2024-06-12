from PIL import Image
import numpy as np

def run_mbrs(image_path1, image_path2, output_path):
    # Open the input images
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)
    
    # Ensure both images have the same mode and size
    if image1.mode != image2.mode:
        raise ValueError("Images must have the same mode.")
    if image1.size != image2.size:
        raise ValueError("Images must have the same size.")
    
    # Convert images to numpy arrays
    array1 = np.array(image1)
    array2 = np.array(image2)
    
    # Calculate the absolute difference
    difference = np.abs(array1 - array2)
    
    # Convert the difference array back to an image
    diff_image = Image.fromarray(difference.astype(np.uint8))
    
    # Save the output image
    diff_image.save(output_path)
    print(f"Difference image saved to {output_path}")
