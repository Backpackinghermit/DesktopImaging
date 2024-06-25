# image_processing.py

import os
from PIL import Image
from utils.image_registration import perform_image_registration
from utils.IRFC import run_IRFC_vill
from utils.irfcirr import run_IRFC_irr
from utils.uvfc import run_UVFC
from gui.image_viewer import create_multiband_tiff
import numpy as np

def process_images(selected_images, checkboxes, registered_images, display_image_callback, update_tab_visibility_callback):
    output_path = "output"  # Define the output path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    vis_image_path = selected_images.get("Vis")
    
    if vis_image_path:
        try:
            # Load and save the visible image
            vis_image = Image.open(vis_image_path)
            output_vis_image_path = os.path.join(output_path, f"Vis_{os.path.basename(vis_image_path)}")
            vis_image.save(output_vis_image_path)
            registered_images["Vis"] = output_vis_image_path  # Update for consistency
            display_image_callback(output_vis_image_path, 'image_label_vis')
        except FileNotFoundError:
            raise FileNotFoundError(f"Visible image not found at: {vis_image_path}")

    if checkboxes['checkbox1']:
        for image_type, image_path in selected_images.items():
            if image_type != "Vis" and image_path:
                try:
                    registered_image_path = perform_image_registration(
                        vis_image_path, image_path, output_path, image_type
                    )
                    if registered_image_path:
                        registered_images[f"Aligned {image_type}"] = registered_image_path
                    else:
                        print(f"Image registration failed for {image_type}")
                except Exception as e:
                    print(f"Image registration failed for {image_type}: {str(e)}")
                    
    if checkboxes['checkbox2']:
        if checkboxes['checkbox1']:
            viil_image_path = registered_images.get("Aligned VIIL")
        else:
            viil_image_path = selected_images.get("VIIL")

        output_path_irfc, error_message = run_IRFC_vill(vis_image_path, viil_image_path, output_path)
        
        if output_path_irfc:
            display_image_callback(output_path_irfc, 'image_label_irfc')
        else:
            print(f"Error during IRFC: {error_message}")
                            
    if checkboxes['checkbox3']:
        if checkboxes['checkbox1']:
            irr_image_path = registered_images.get("Aligned IRR")
        else:
            irr_image_path = selected_images.get("IRR")
            
        output_path_irfcirr, error_message = run_IRFC_irr(vis_image_path, irr_image_path, output_path)
        if output_path_irfcirr:
            display_image_callback(output_path_irfcirr, 'image_label_irfcirr')
        else:
            print(f"Error during IRFC: {error_message}")
                    
    if checkboxes['checkbox4']:
        if checkboxes['checkbox1']:
            uvr_image_path = registered_images.get("Aligned UVR")
        else:
            uvr_image_path = selected_images.get("UVR")
            
        output_path_uvr, error_message = run_UVFC(vis_image_path, uvr_image_path, output_path)
        if output_path_uvr:
            display_image_callback(output_path_uvr, 'image_label_uvfc')
        else:
            print(f"Error during IRFC: {error_message}")

    if checkboxes['checkbox6']:
        vis_image_path = output_vis_image_path  # Ensure this is defined or passed correctly
        output_tiff_path = os.path.join(output_path, f"multiband_{os.path.basename(vis_image_path)}")
        try:
            create_multiband_tiff(registered_images, output_tiff_path, output_vis_image_path)
            display_image_callback(output_tiff_path, 'image_label_multiband')
        except Exception as e:
            return f"Failed to separate RGB channels: {str(e)}"

    update_tab_visibility_callback()
