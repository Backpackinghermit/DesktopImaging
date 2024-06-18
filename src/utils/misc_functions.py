from PyQt5.QtWidgets import QFileDialog, QScrollArea, QWidget, QVBoxLayout, QLabel, QGridLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from gui.image_viewer import scroll_areas, init_tab


def update_tab_visibility(self):
        label_mapping = {
            "IRR": self.image_label_irr,
            "VIIL": self.image_label_viil,
            "UVR": self.image_label_uvr,
            "UVF": self.image_label_uvf,
            "Aligned IRR": self.image_label_Regirr,
            "Aligned VIIL": self.image_label_Regviil,
            "Aligned UVR": self.image_label_Reguvr,
            "Aligned UVF": self.image_label_Reguvf,
            "Vis": self.image_label_vis,
            "IRFC": self.image_label_irfc,
            "IRFCIRR": self.image_label_irfcirr,
            "UVFC": self.image_label_uvfc,
            "Multiband": self.image_label_multiband
        }

        # Hide all tabs initially
        for index in range(self.tabs.count()):
            self.tabs.setTabVisible(index, False)

        # Show tabs with images
        for image_type, image_path in self.registered_images.items():
            if image_path:
                image_label = label_mapping.get(image_type)
                if image_label:
                    self.display_image_in_tab(image_path, image_label)  # Populate the existing label
                    # Make the tab visible
                    tab_index = self.tabs.indexOf(image_label.parent().parent())
                    self.tabs.setTabVisible(tab_index, True)
                else:
                    print(f"Image label not found for {image_type}")

        # Ensure specific tabs are shown based on checkboxes and image presence
        self.tabs.setTabVisible(self.tabs.indexOf(self.image_label_align.parent().parent()), self.checkbox1.isChecked() and self.image_label_align.pixmap() is not None)
        self.tabs.setTabVisible(self.tabs.indexOf(self.image_label_irfc.parent().parent()), self.checkbox2.isChecked() and self.image_label_irfc.pixmap() is not None)
        self.tabs.setTabVisible(self.tabs.indexOf(self.image_label_irfcirr.parent().parent()), self.checkbox3.isChecked() and self.image_label_irfcirr.pixmap() is not None)
        self.tabs.setTabVisible(self.tabs.indexOf(self.image_label_uvfc.parent().parent()), self.checkbox4.isChecked() and self.image_label_uvfc.pixmap() is not None)
        self.tabs.setTabVisible(self.tabs.indexOf(self.image_label_multiband.parent().parent()), self.checkbox6.isChecked() and self.image_label_multiband.pixmap() is not None)
    
def thumbnail_clicked(self, image_path, image_type):
        # Check if a tab already exists for this image_type
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == image_type:
                self.tabs.setCurrentIndex(i)  # Switch to the tab if it already exists
                return

        # If no tab exists, create a new tab
        new_label = ZoomableLabel()
        new_label.setStyleSheet("background-color: white;")
        self.display_image_in_tab(image_path, new_label)

        scroll_area = QScrollArea()
        scroll_area.setWidget(new_label)
        scroll_area.setWidgetResizable(True)

        self.tabs.addTab(scroll_area, image_type)
        self.tabs.setCurrentWidget(scroll_area)
   
def clear_tabs(self):
        # Set visibility of all tabs except "Processed Images" to False
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) != "Processed Images":
                self.tabs.setTabVisible(i, False)
        
        # Set visibility of "Processed Images" tab to True
        self.tabs.setTabVisible(self.tabs.indexOf(self.scroll_areas["Processed Images"]), True)

def init_tab(label, title):
        if title not in scroll_areas:
            scroll_areas[title] = QScrollArea()
            scroll_areas[title].setWidgetResizable(True)
            scroll_areas[title].setWidget(label)

        tabs.addTab(scroll_areas[title], title)
        
        # Set visibility for the "Processed Images" tab
        if title == "Processed Images":
            tabs.setCurrentWidget(scroll_areas[title])
        else:
            tabs.setTabVisible(tabs.indexOf(scroll_areas[title]), False)
   
def open_image_and_display_thumbnail(self, image_type):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, f"Open {image_type} Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        if file_path:
            if self.selected_images[image_type]:
                # Remove the existing image path for the given image type
                del self.selected_images[image_type]

                # Remove the existing thumbnail widget from the layout
                item = self.thumbnail_container.itemAt(0)
                if item:
                    widget = item.widget()
                    if widget:
                        widget.deleteLater()

            # Update the selected_images dictionary with the new image path
            self.selected_images[image_type] = file_path
            # Add the thumbnail for the newly selected image
            self.add_thumbnail(image_type, file_path)
            # Reorder the thumbnails as needed
            self.reorder_thumbnails()

def add_thumbnail(self, image_type, image_path):
        widget = QWidget()
        layout = QVBoxLayout()
        

        # Create QLabel for image
        image_label = QLabel()
        pixmap = QPixmap(image_path)
        image_label.setPixmap(pixmap.scaled(150, 150, Qt.KeepAspectRatio))
        layout.addWidget(image_label)

        # Create QLabel for text
        text_label = QLabel(image_type)
        
        layout.addWidget(text_label)

        widget.setLayout(layout)
        self.thumbnail_container.addWidget(widget)
        

def reorder_thumbnails(self):
        for i in reversed(range(self.thumbnail_container.count())):
            widget = self.thumbnail_container.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        for image_type in self.image_order:
            # Handle selected_images
            file_path = self.selected_images.get(image_type)
            if file_path:
                self.add_thumbnail(image_type, file_path)

            # Handle registered_images
            registered_path = self.registered_images.get(image_type)
            if registered_path:
                self.add_thumbnail(image_type, registered_path)

    
def display_image_in_tab(self, image_path, label):
        if image_path:
            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                print(f"Failed to load image: {image_path}")
                return

            # Set the pixmap to the ZoomableLabel with initial scaling
            label.set_initial_pixmap(pixmap)
            label.update()
            label.setAlignment(Qt.AlignCenter)
            
def resizeEvent(self, event):
            super().resizeEvent(event)
            for i in range(self.tabs.count()):
                tab = self.tabs.widget(i)
                if tab:
                    image_label = tab.widget()
                    if image_label and image_label.pixmap():
                        scaled_pixmap = image_label.pixmap().scaled(image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        image_label.setPixmap(scaled_pixmap)

def clear_images(self):
        # Reset all selected images to None
        for image_type in self.selected_images:
            self.selected_images[image_type] = None
            self.registered_images = {}
            self.update_thumbnails()
            self.clear_tabs()
            self.update_tab_visibility
    
def thumbnail_clicked(self, event, image_path, image_type):
        if image_path:
            self.selected_images[image_type] = image_path
            self.update_thumbnails()
     # Check if a tab already exists for this image_type
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == image_type:
                self.tabs.setCurrentIndex(i)  # Switch to the tab if it already exists
                return

        # If no tab exists, create a new tab
        new_label = QLabel()
        new_label.setStyleSheet("background-color: white;")
        self.display_image_in_tab(image_path, new_label)
        
        scroll_area = QScrollArea()
        scroll_area.setWidget(new_label)
        scroll_area.setWidgetResizable(True)
        
        self.tabs.addTab(scroll_area, image_type)
        self.tabs.setCurrentWidget(scroll_area)
        self.update_tab_visibility()

def save_multiband_tiff(self):
        if not all(self.registered_images.values()):
            QMessageBox.warning(self, "Warning", "Please process all images before saving.")
            return

        options = QFileDialog.Options()
        output_path, _ = QFileDialog.getSaveFileName(self, "Save Multi-Band TIFF", "", "TIFF Files (*.tiff);;All Files (*)", options=options)
        if output_path:
            try:
                create_multiband_tiff(self.registered_images, output_path, self.selected_images["Vis"])  # Pass visible image path
                QMessageBox.information(self, "Success", "Multi-Band TIFF saved successfully.")
            except Exception as e:  # Catch all exceptions
                QMessageBox.critical(self, "Error", f"Failed to create Multi-Band TIFF: {str(e)}")

def update_thumbnails(self):
        # Clear existing thumbnails
        for i in reversed(range(self.thumbnail_container.count())):
            widget = self.thumbnail_container.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        # Add thumbnails for each image type in selected_images
        for image_type in self.selected_images:
            file_path = self.selected_images[image_type]
            if file_path:
                self.add_thumbnail(image_type, file_path)

        # Add thumbnails for each image type in registered_images
        for image_type in self.registered_images:
            file_path = self.registered_images[image_type]
            if file_path:
                self.add_thumbnail(image_type, file_path)

def save_all_images(self):
        """Saves all images in the output folder to a user-selected directory and then clears the output folder."""

        output_folder = "output"  # Your output folder path (replace if it's different)
        if not os.path.exists(output_folder):
            QMessageBox.warning(self, "Warning", "Output folder not found!")
            return

        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if not directory:
            return  # User canceled the dialog

        for filename in os.listdir(output_folder):
            if filename.endswith((".png", ".jpg", ".jpeg", ".tiff")):  # Add more extensions as needed
                source_path = os.path.join(output_folder, filename)
                destination_path = os.path.join(directory, filename)
                try:
                    shutil.copy2(source_path, destination_path)  # Use shutil.copy2 to preserve metadata
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to save {filename}: {e}")

        # Clear the output folder
        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to clear output folder: {e}")
        QMessageBox.about(self, "Done", "Images Saved")
