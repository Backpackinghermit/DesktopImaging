import numpy as np
from PIL import Image
import tifffile as tiff
import os

import os
import subprocess

def create_multiband_tiff_IM(vis_image_path, output_dir):
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

    
    if result.returncode != 0:
        print("Error occurred during RGB separation:", result.stderr)
    else:
        print("RGB channels saved")

        
        
def combine_multiband_tiff(output_path, red_channel_path, green_channel_path, blue_channel_path):
    output_tiff = os.path.join(output_path, "multiband.tiff")

    # Hardcoded paths to registered images (replace with your actual paths)
    registered_irr_path = os.path.join(output_path, "registered_IRR.png")
    registered_uvr_path = os.path.join(output_path, "registered_UVR.png")
    registered_uvf_path = os.path.join(output_path, "registered_UVF.png")
    registered_viil_path = os.path.join(output_path, "registered_VIIL.png")

    # Construct the command to combine all channels
    command = [
        "magick", "-depth", "8",
        red_channel_path, green_channel_path, blue_channel_path,
        registered_irr_path, registered_uvr_path, registered_uvf_path, registered_viil_path,
        "-combine", output_tiff
    ] 

    # Execute the command
    print(command)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error occurred:", result.stderr)
    else:
        print(f"Multiband TIFF file saved at: {output_tiff}")

def create_multiband_tiff(registered_images, output_tiff_path, vis_image_path):
    image_types = ["IRR", "VIIL", "UVR", "UVF"]

    try:
        vis_image = np.array(Image.open(vis_image_path))

        # Check if visible image is RGB
        if vis_image.ndim == 3 and vis_image.shape[2] == 3:
            # Transpose to (channels, height, width) for correct TIFF format
            images = list(np.moveaxis(vis_image, -1, 0))  # Move channels to the front and convert to list
            print("Loaded visible image with shape:", vis_image.shape)
        else:
            raise ValueError("Visible image must have 3 channels (RGB)")

        # Load and resize registered images
        for image_type in image_types:
            image_path = registered_images.get(image_type)
            if image_path is not None:
                try:
                    image = np.array(Image.open(image_path))
                    # Resize if dimensions don't match
                    if image.shape[:2] != vis_image.shape[:2]:
                        image = Image.fromarray(image)
                        image = image.resize(vis_image.shape[:2])
                        image = np.array(image)
                    
                    images.append(image)
                    print(f"Loaded {image_type} image with shape:", image.shape)

                except FileNotFoundError:
                    print(f"Warning: Image of type '{image_type}' not found at: {image_path}")

        # Ensure at least the visible image is present
        if len(images) < 3:  # Check for at least 3 channels (RGB)
            raise ValueError("Not enough channels found for the visible image")

        print("All images have consistent dimensions:", images[0].shape)  # Verify after resizing

        # Stack images along the third axis (channels axis) - already in correct shape
        multiband_image = np.stack(images, axis=0)  
        print("Multiband image shape:", multiband_image.shape)

        # Save the multiband image to a TIFF file
        try:
            tiff.imwrite(output_tiff_path, multiband_image)
            print(f"Multiband TIFF file saved at: {output_tiff_path}")
        except PermissionError:
            print(f"Permission error: Unable to save file at {output_tiff_path}.")
        except FileNotFoundError:
            print(f"File not found error: Unable to save file at {output_tiff_path}. Ensure the directory exists.")
        except Exception as e:
            print(f"An error occurred while saving the TIFF file: {str(e)}")
    except ValueError as e:
        print(f"Value Error in create multiband tiff: {str(e)}")

