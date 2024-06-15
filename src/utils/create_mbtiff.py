import numpy as np
from PIL import Image
import tifffile as tiff
import os

import os
import subprocess

def create_multiband_tiff(registered_images, output_tiff_path, vis_image_path):
    image_types = ["IRR", "VIIL", "UVR", "UVF"]
    images = []

    try:
        vis_image = np.array(Image.open(vis_image_path))

        # Check if visible image is RGB
        if vis_image.ndim == 3 and vis_image.shape[2] == 3:
            # Separate the RGB channels
            red_channel = vis_image[:, :, 0]
            green_channel = vis_image[:, :, 1]
            blue_channel = vis_image[:, :, 2]

            images.extend([red_channel, green_channel, blue_channel])
            print("Loaded visible image with shape:", vis_image.shape)
        else:
            raise ValueError("Visible image must have 3 channels (RGB)")

        # Load and resize registered images
        for image_type in image_types:
            image_path = registered_images.get(image_type)
            if image_path is not None:
                try:
                    image = np.array(Image.open(image_path).convert('L'))
                    # Resize if dimensions don't match
                    if image.shape != vis_image.shape[:2]:
                        image = Image.fromarray(image)
                        image = image.resize((vis_image.shape[1], vis_image.shape[0]))  # Ensure width, height match
                        image = np.array(image)

                    images.append(image)
                    print(f"Loaded {image_type} image with shape:", image.shape)

                except FileNotFoundError:
                    print(f"Warning: Image of type '{image_type}' not found at: {image_path}")

        print("All images have consistent dimensions:", images[0].shape)  # Verify after resizing

        # Stack images along the first axis (channels axis)
        multiband_image = np.stack(images, axis=0)
        print("Multiband image shape:", multiband_image.shape)

        # Save the multiband image to a TIFF file
        try:
            tiff.imwrite(output_tiff_path, multiband_image, photometric='minisblack')
            print(f"Multiband TIFF file saved at: {output_tiff_path}")
        except PermissionError:
            print(f"Permission error: Unable to save file at {output_tiff_path}.")
        except FileNotFoundError:
            print(f"File not found error: Unable to save file at {output_tiff_path}. Ensure the directory exists.")
        except Exception as e:
            print(f"An error occurred while saving the TIFF file: {str(e)}")
    except ValueError as e:
        print(f"Value Error in create multiband tiff: {str(e)}")