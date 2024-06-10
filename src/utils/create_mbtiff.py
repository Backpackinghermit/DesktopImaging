import os
import subprocess

def create_multiband_tiff(registered_images, output_path, vis_image_path):
    image_types = ["IRR", "VIIL", "UVR", "UVF"]

    # Construct the command
    command = ["convert", vis_image_path]
    
    # Add the registered images to the command
    for image_type in image_types:
        image_path = registered_images.get(image_type)
        if image_path:
            command.append(image_path)

    # Specify the channel combining logic
    num_channels = len(image_types) + 3  # 3 for RGB + number of registered images
    channel_fx = "; ".join([f"{i},{i},{i}" for i in range(num_channels)])
    command.extend(["-channel-fx", channel_fx, output_path])

    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error occurred:", result.stderr)
    else:
        print(f"Multiband TIFF file saved at: {output_path}")