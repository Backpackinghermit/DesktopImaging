import numpy as np
from PIL import Image
import tifffile as tiff
import os

import os
import subprocess

def create_multiband_tiff(vis_image_path, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Paths for separated RGB channels
    red_channel_path = os.path.join(output_dir, "R.png")
    green_channel_path = os.path.join(output_dir, "G.png")
    blue_channel_path = os.path.join(output_dir, "B.png")

    # Construct the separation command
    separate_command = [
        "magick", vis_image_path, "-separate", 
        os.path.join(output_dir, "channel_%d.png")
    ]
    
    # Execute the separation command
    result = subprocess.run(separate_command, capture_output=True, text=True)
    
    # Rename the channels to R.png, G.png, and B.png
    os.rename(os.path.join(output_dir, "channel_0.png"), red_channel_path)
    os.rename(os.path.join(output_dir, "channel_1.png"), green_channel_path)
    os.rename(os.path.join(output_dir, "channel_2.png"), blue_channel_path)
    
    print(f"RGB channels saved as {red_channel_path}, {green_channel_path}, and {blue_channel_path}")
    
    # Call the combine_multiband_tiff function with additional parameters
    
    if result.returncode != 0:
        print("Error occurred during RGB separation:", result.stderr)
    else:
        print("Multiband TIFF saved")

        
        
def combine_multiband_tiff(registered_images, output_path, red_channel_path, green_channel_path, blue_channel_path):
    image_types = ["IRR", "VIIL", "UVR", "UVF"]

    # Construct the command
    command = ["magick", red_channel_path, green_channel_path, blue_channel_path]
    
    # Add the registered images to the command
    for image_type in image_types:
        image_path = registered_images.get(image_type)
        if image_path:
            command.append(image_path)

    # Combine the images into a multiband TIFF
    command.extend(["-combine", "-depth", "8", output_path])  # Assuming 8-bit depth

    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error occurred:", result.stderr)
    else:
        print(f"Multiband TIFF file saved at: {output_path}")

def create_multiband_tiff_py(registered_images, output_path, vis_image_path):
    image_types = ["IRR", "VIIL", "UVR", "UVF"]

    # Load visible image (always required)
    try:
        vis_image = np.array(Image.open(vis_image_path))
        if vis_image.ndim != 3 or vis_image.shape[2] != 3:
            raise ValueError("Visible image must have 3 channels (RGB)")
        # Split visible image into RGB channels and append to list
        images = list(np.moveaxis(vis_image, -1, 0))  # Move channels to front and convert to list
        print("Loaded visible image with shape:", vis_image.shape)
    except FileNotFoundError:
        raise FileNotFoundError(f"Visible image not found at: {vis_image_path}")

    # Load registered images (optional)
    for image_type in image_types:
        image_path = registered_images.get(image_type)
        if image_path is not None:
            try:
                image = np.array(Image.open(image_path))
                if image.shape[:2] != vis_image.shape[:2]:
                    raise ValueError(f"Image dimensions for {image_type} do not match visible image dimensions")
                images.append(image)
                print(f"Loaded {image_type} image with shape:", image.shape)
            except FileNotFoundError:
                print(f"Warning: Image of type '{image_type}' not found at: {image_path}")

    # Ensure at least the visible image is present
    if len(images) < 3:  # Check for at least 3 channels (RGB)
        raise ValueError("Not enough channels found for the visible image")

    # Ensure all images have the same dimensions
    base_shape = images[0].shape
    for img in images:
        if img.shape != base_shape:
            raise ValueError("All images must have the same dimensions")

    print("All images have consistent dimensions:", base_shape)

    # Stack images as separate bands (new axis)
    multiband_image = np.stack(images, axis=0)
    print("Multiband image shape before transpose:", multiband_image.shape)

    # Transpose the image to correct the shape
    multiband_image = np.transpose(multiband_image, (1, 2, 0))
    print("Multiband image shape after transpose:", multiband_image.shape)

    # Save the multiband image to a TIFF file
    tiff.imwrite(output_path, multiband_image)
    print(f"Multiband TIFF file saved at: {output_path}")

