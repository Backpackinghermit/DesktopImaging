from PIL import Image
import os
import subprocess

def run_IRFC_irr_IM(image_path, irr_image_path, output_folder="processed_images"):
    try:
        filename = os.path.basename(image_path)
        processed_filename = 'IRFCirr_processed_' + filename
        output_path = os.path.join(output_folder, processed_filename)

        os.makedirs(output_folder, exist_ok=True)

        # ImageMagick command for channel swapping (updated for multiple input images)
        # Using list format for better argument handling
        command = [
            'magick',
            image_path,
            '-auto-orient',
            irr_image_path,
            '-auto-orient',
            '-channel', 'rgba',
            '-separate',
            '-swap', '1,2',
            '-swap', '0,1',
            '-swap', '0,3',
            '-channel', 'rgb',
            '-combine',
            os.path.join(output_path)
        ]

        # Run ImageMagick with improved error handling
        result = subprocess.run(
            command,
            check=True,  
            capture_output=True,  
            text=True 
        )

        if result.returncode == 0:
            return output_path, None  
        else:
            error_message = result.stderr  # Capture stderr for detailed errors
            return None, error_message  

    except subprocess.CalledProcessError as e:
        return None, e.stderr  # Return the error message from ImageMagick

    except Exception as e:
        return None, str(e)   # Return a string representation of other errors

def run_IRFC_irr(image_path, irr_image_path, output_folder="processed_images"):
    try:
        filename = os.path.basename(image_path)
        processed_filename = "IRFC-irr" + filename
        output_path = os.path.join(output_folder, processed_filename)
        os.makedirs(output_folder, exist_ok=True)

        # Load the images using Pillow
        with Image.open(image_path) as img, Image.open(irr_image_path) as irr_img:
            img = img.convert('RGB')
            irr_img = irr_img.convert('RGB')
            
            # Split images into RGB channels
            r, g, b = img.split()
            _, irr_g, _ = irr_img.split()  # Only use the green channel from UVR

            # Perform channel swapping
            swapped_image = Image.merge("RGB", (irr_g, r, g)) 

            # Save the processed image
            swapped_image.save(output_path)

        return output_path, None  # Success

    except (FileNotFoundError, ValueError) as e:
        return None, str(e)  # Return error message

    except Exception as e:
        return None, str(e)  # Catch any other unexpected errors
