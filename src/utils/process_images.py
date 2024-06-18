from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
import os
from PIL import Image
from utils.image_registration import perform_image_registration



def process_images(self):
        output_path = "output"  # Define the output path
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        self.tabs.setTabVisible(self.tabs.indexOf(self.scroll_areas["Processed Images"]), False)

        vis_image_path = self.selected_images.get("Vis")
        
        if vis_image_path:
            try:
                # Load and save the visible image
                vis_image = Image.open(vis_image_path)
                output_vis_image_path = os.path.join(output_path, f"Vis_{os.path.basename(vis_image_path)}")
                vis_image.save(output_vis_image_path)
                self.registered_images["Vis"] = output_vis_image_path  # Update for consistency
                self.display_image_in_tab(output_vis_image_path, self.image_label_vis)
            except FileNotFoundError:
                raise FileNotFoundError(f"Visible image not found at: {vis_image_path}")

        if self.checkbox1.isChecked():
            for image_type, image_path in self.selected_images.items():
                if image_type != "Vis" and image_path:
                    try:
                        # Get the registered image path from the function
                        registered_image_path = perform_image_registration(
                            vis_image_path, image_path, output_path, image_type
                        )
                        
                        if registered_image_path:  # Check if registration was successful
                            self.registered_images[f"Aligned {image_type}"] = registered_image_path
                            print(self.registered_images)
                        else:
                            print(f"Image registration failed for {image_type}")
                    except Exception as e:
                        print(f"Image registration failed for {image_type}: {str(e)}")
                        
        if self.checkbox2.isChecked():
            if self.checkbox1.isChecked():
                viil_image_path = self.registered_images.get("Aligned VIIL")
            else:
                viil_image_path = self.selected_images.get("VIIL")

            # Debug print to check the paths
            print(f"vis_image_path: {vis_image_path}")
            print(f"viil_image_path: {viil_image_path}")

            output_path_irfc, error_message = run_IRFC_vill(vis_image_path, viil_image_path, output_path)
            
            if output_path_irfc:
                self.display_image_in_tab(output_path_irfc, self.image_label_irfc)
            else:
                print(f"Error during IRFC: {error_message}")
                                
        if self.checkbox3.isChecked():
            if self.checkbox1.isChecked():
                irr_image_path = self.registered_images.get("Aligned IRR")
            else:
                irr_image_path = self.selected_images.get("IRR")
                
            output_path_irfcirr, error_message = run_IRFC_irr(vis_image_path, irr_image_path, output_path)
            if output_path_irfcirr:
                self.display_image_in_tab(output_path_irfcirr, self.image_label_irfcirr)
            else:
                print(f"Error during IRFC: {error_message}")
                        
        if self.checkbox4.isChecked():
            if self.checkbox1.isChecked():
                uvr_image_path = self.registered_images.get("Aligned UVR")
            else:
                uvr_image_path = self.selected_images.get("UVR")
                
            output_path_uvr, error_message = run_UVFC(vis_image_path, uvr_image_path, output_path)
            if output_path_uvr:
                self.display_image_in_tab(output_path_uvr, self.image_label_uvfc)
            else:
                print(f"Error during IRFC: {error_message}")

        if self.checkbox6.isChecked():
            vis_image_path = output_vis_image_path  # Ensure this is defined or passed correctly
            output_tiff_path = os.path.join(output_path, f"multiband_{os.path.basename(vis_image_path)}")
            try:
                create_multiband_tiff(self.registered_images, output_tiff_path, output_vis_image_path)
                self.display_image_in_tab(output_tiff_path, self.image_label_multiband)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to separate RGB channels: {str(e)}")
        
        self.update_tab_visibility()
