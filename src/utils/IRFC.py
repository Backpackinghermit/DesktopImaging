import os
import subprocess

def run_IRFC(image_path, viil_image_path, output_folder="processed_images"):
    try:
        filename = os.path.basename(image_path)
        processed_filename = 'IRFC_processed_' + filename
        output_path = os.path.join(output_folder, processed_filename)

        os.makedirs(output_folder, exist_ok=True)

        # ImageMagick command for channel swapping (updated for multiple input images)
        # Using list format for better argument handling
        command = [
            'magick',
            image_path,
            '-auto-orient',
            viil_image_path,
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
