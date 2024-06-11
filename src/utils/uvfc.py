from PIL import Image
import os

def run_UVFC(image_path, uvr_image_path, output_folder="processed_images"):
    try:
        filename = os.path.basename(image_path)
        processed_filename = "UVFC_processed_" + filename
        output_path = os.path.join(output_folder, processed_filename)
        os.makedirs(output_folder, exist_ok=True)

        # Load the images using Pillow
        with Image.open(image_path) as img, Image.open(uvr_image_path) as uvr_img:
            img = img.convert('RGB')
            uvr_img = uvr_img.convert('RGB')
            
            # Split images into RGB channels
            r, g, b = img.split()
            _, uvr_g, _ = uvr_img.split()  # Only use the green channel from UVR

            # Perform channel swapping
            swapped_image = Image.merge("RGB", (g, b, uvr_g))  # g -> R, b -> G, uvr_g -> B

            # Save the processed image
            swapped_image.save(output_path)

        return output_path, None  # Success

    except (FileNotFoundError, ValueError) as e:
        return None, str(e)  # Return error message

    except Exception as e:
        return None, str(e)  # Catch any other unexpected errors
