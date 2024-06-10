import os
import subprocess
from PyQt5.QtWidgets import (QMainWindow, QAction, QFileDialog, QLabel, QScrollArea, QMessageBox,
                             QWidget, QHBoxLayout, QVBoxLayout, QFrame, QPushButton, QApplication, 
                             QCheckBox, QSizePolicy, QTabWidget, QStyle )
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import numpy as np
import PIL
from PIL import Image

# Import the image registration function
from utils.image_registration import perform_image_registration
from utils.IRFC import run_IRFC  # Import the IRFC function
from utils.uvfc import run_UVFC
from gui import image_viewer
from utils.create_mbtiff import create_multiband_tiff, combine_multiband_tiff




class ImageApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.title = "Conservation Image Processing"
        self.setGeometry(100, 100, 1500, 800)
        self.scroll_areas = {}  # Store scroll areas
        

        self.pixmap = None

        self.selected_images = {"Vis": None, "IRR": None, "VIIL": None, "UVR": None, "UVF": None}
        self.registered_images = {"Vis": None, "IRR": None, "VIIL": None, "UVR": None, "UVF": None}

        self.image_order = ["UVR", "UVF", "Vis", "IRR", "VIIL"]

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.title)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
    
        main_layout = QVBoxLayout(central_widget)
        central_widget.setStyleSheet("background-color: #d3d3d3;")

        top_row = QFrame()
        top_row.setFrameShape(QFrame.StyledPanel)
        top_row_layout = QHBoxLayout(top_row)
        # Create a scroll area for the image preview
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: white; 
                border: none;
                border-radius: 10px;
            }
        """)
        # Create a content widget for the scroll area
        content_widget = QWidget()
        content_widget.setStyleSheet("""
            QWidget {
                background-color: white; 
                border-top-left-radius: 10px; 
                border-top-right-radius: 10px;
                border-bottom-left-radius: 10px;
                border-bottom-right-radius: 10px;
            }
        """)
         # Layout for the content widget
        content_layout = QVBoxLayout(content_widget)

        # Add the image label to the content layout
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(self.image_label)
        
        

        section1 = QVBoxLayout()
        button1 = QPushButton("Vis")
        button2 = QPushButton("IRR")
        button3 = QPushButton("VIIL")
        button4 = QPushButton("UVR")
        button5 = QPushButton("UVF")

        button1.clicked.connect(lambda: self.open_image_and_display_thumbnail("Vis"))
        button2.clicked.connect(lambda: self.open_image_and_display_thumbnail("IRR"))
        button3.clicked.connect(lambda: self.open_image_and_display_thumbnail("VIIL"))
        button4.clicked.connect(lambda: self.open_image_and_display_thumbnail("UVR"))
        button5.clicked.connect(lambda: self.open_image_and_display_thumbnail("UVF"))

        section1.addWidget(button4)
        section1.addWidget(button5)
        section1.addWidget(button1)
        section1.addWidget(button2)
        section1.addWidget(button3)
        
        button1.setStyleSheet("border-radius: 5px; background-color: white; padding: 5px;")
        button2.setStyleSheet("border-radius: 5px; background-color: white; padding: 5px;")
        button3.setStyleSheet("border-radius: 5px; background-color: white; padding: 5px;")
        button4.setStyleSheet("border-radius: 5px; background-color: white; padding: 5px;")
        button5.setStyleSheet("border-radius: 5px; background-color: white; padding: 5px;")
        

        section1_widget = QWidget()
        section1_widget.setLayout(section1)
        section1_widget.setFixedWidth(100)
        section1_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        top_row_layout.addWidget(section1_widget)

        section2 = QWidget()
        section2_layout = QHBoxLayout(section2)
        self.thumbnail_container = QHBoxLayout()
        section2_layout.addLayout(self.thumbnail_container)
        top_row_layout.addWidget(section2)
        

        main_layout.addWidget(top_row)
        
        self.checkbox4 = QCheckBox("UVFC")
        self.checkbox4.setChecked(False)

        top_row_widget = QWidget()
        top_row_widget.setLayout(top_row_layout)
        scroll_area.setWidget(top_row_widget)
        scroll_area.setMaximumHeight(230)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: #d3d3d3; 
                border: 0px;
            }
            QScrollBar:horizontal {
                border: 1px solid #999999;
                background: #d3d3d3;
                height: 8px;
                margin: 0px 0px 0px 0px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background: #888888;
                min-width: 20px;
                border-radius: 10px;
            }
            QScrollBar::add-line:horizontal {
                background: #c2c2c2;
                width: 0px;
                subcontrol-position: right;
                subcontrol-origin: margin;
                border-radius: 6px;
            }
            QScrollBar::sub-line:horizontal {
                background: #c2c2c2;
                width: 0px;
                subcontrol-position: left;
                subcontrol-origin: margin;
                border-radius: 6px;
            }
            QScrollBar::left-arrow:horizontal, QScrollBar::right-arrow:horizontal {
                border: 2px solid grey;
                width: 3px;
                height: 3px;
                background: white;
                border-radius: 6px;
            }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: none;
            }
            QScrollBar:vertical {
                border: 1px solid #999999;
                background: #d3d3d3;
                width: 8px;
                margin: 0px 0px 0px 0px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #888888;
                min-height: 20px;
                border-radius: 6px;
            }
            QScrollBar::add-line:vertical {
                background: #c2c2c2;
                height: 0px;
                subcontrol-position: bottom;
                subcontrol-origin: margin;
                border-radius: 6px;
            }
            QScrollBar::sub-line:vertical {
                background: #c2c2c2;
                height: 0 px;
                subcontrol-position: top;
                subcontrol-origin: margin;
                border-radius: 6px;
            }
            QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
                border: 2px solid grey;
                width: 3px;
                height: 3px;
                background: white;
                border-radius: 6px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        
        """)

        main_layout.addWidget(scroll_area)

        bottom_row = QFrame()
        bottom_row.setFrameShape(QFrame.StyledPanel)
        bottom_row_layout = QHBoxLayout(bottom_row)
        bottom_row.setStyleSheet("background-color: #d3d3d3; border: 0px; border-radius: 20px")
               

        self.tabs = QTabWidget()
        self.image_label_align = QLabel()
        self.image_label_irfc = QLabel()
        self.image_label_irfcirr = QLabel()
        self.image_label_uvr = QLabel()
        self.image_label_processed = QLabel()
        self.image_label_processed.setStyleSheet("background-color: white; border-top-left-radius: 0px;")
        self.image_label_align.setStyleSheet("background-color: white; border-top-left-radius: 0px;")
        self.image_label_irfc.setStyleSheet("background-color: white; border-top-left-radius: 0px;")
        self.image_label_irfcirr.setStyleSheet("background-color: white; border-top-left-radius: 0px;")
        self.image_label_uvr.setStyleSheet("background-color: white; border-top-left-radius: 0px;")
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 0;
            }
            QTabBar::tab {
                border: 5;
                background: #bfbfbf;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                min-width: 8ex;
                padding-right: 15px;
                padding-left: 15px;
                padding-top: 5px;
                padding-bottom: 5px;
            }
            QTabBar::tab:selected, QTabBar::tab:hover {
                background: white;
            }
            QTabBar::tab:selected {
                border-color: #9b9b9b;
                border-bottom-color: #f0f0f0; /* same as pane color */
            }
        """)

        self.init_tab(self.image_label_align, "Align")
        self.init_tab(self.image_label_irfc, "IRFC (VIIL)")
        self.init_tab(self.image_label_irfcirr, "IRFC (IRR)")
        self.init_tab(self.image_label_uvr, "UVFC")
        self.init_tab(self.image_label_processed, "Processed Images")
    

        bottom_row_layout.addWidget(self.tabs)
        
        right_column = QFrame()
        right_column.setFrameShape(QFrame.StyledPanel)
        right_column_layout = QVBoxLayout(right_column)
        right_column.setStyleSheet("border: 0px;")

        self.checkbox1 = QCheckBox("Align")
        self.checkbox2 = QCheckBox("IRFC (VIIL)")
        self.checkbox3 = QCheckBox("IRFC (IRR)")
        self.checkbox4 = QCheckBox("UVFC")
        self.checkbox5 = QCheckBox("MBIS(850-740)")
        self.checkbox6 = QCheckBox("Multiband tiff")

        right_column_layout.addWidget(self.checkbox1)
        right_column_layout.addWidget(self.checkbox2)
        right_column_layout.addWidget(self.checkbox3)
        right_column_layout.addWidget(self.checkbox4)
        right_column_layout.addWidget(self.checkbox5)
        right_column_layout.addWidget(self.checkbox6)

        process_button = QPushButton("Process")
        process_button.setStyleSheet("border-radius: 5px; background-color: white; padding: 10px;")
        process_button.clicked.connect(self.process_images)
        right_column_layout.addWidget(process_button)

        bottom_row_layout.addWidget(right_column)
        main_layout.addWidget(bottom_row)
        
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        edit_menu = menubar.addMenu('Edit')
        # Create clear action
        clear_action = QAction('Clear', self)
        clear_action.triggered.connect(self.clear_images)

        exit_action = QAction('Exit', self)
        save_action = QAction('Save All', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(clear_action)
        file_menu.addAction(exit_action)
        file_menu.addAction(save_action)
        
    def update_tab_visibility(self):
        self.tabs.setTabVisible(0, self.checkbox1.isChecked() and self.image_label_align.pixmap() is not None) 
        self.tabs.setTabVisible(1, self.checkbox2.isChecked() and self.image_label_irfc.pixmap() is not None)
        self.tabs.setTabVisible(2, self.checkbox3.isChecked() and self.image_label_irfcirr.pixmap() is not None)
        self.tabs.setTabVisible(3, self.checkbox4.isChecked() and self.image_label_uvr.pixmap() is not None) # UVR tab visibility
        
        
    def clear_tabs(self):
        # Set visibility of all tabs except "Processed Images" to False
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) != "Processed Images":
                self.tabs.setTabVisible(i, False)
        
        # Set visibility of "Processed Images" tab to True
        self.tabs.setTabVisible(self.tabs.indexOf(self.scroll_areas["Processed Images"]), True)

    def init_tab(self, label, title):
        if title not in self.scroll_areas:
            self.scroll_areas[title] = QScrollArea()
            self.scroll_areas[title].setWidgetResizable(True)
            self.scroll_areas[title].setWidget(label)

        self.tabs.addTab(self.scroll_areas[title], title)
        
        # Set visibility for the "Processed Images" tab
        if title == "Processed Images":
            self.tabs.setCurrentWidget(self.scroll_areas[title])
        else:
            self.tabs.setTabVisible(self.tabs.indexOf(self.scroll_areas[title]), False)
   
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

    def process_images(self):
        irr_image_path = self.selected_images["IRR"]
        vis_image_path = self.selected_images["Vis"]
        viil_image_path = self.selected_images["VIIL"]
        uvr_image_path = self.selected_images["UVR"]
        output_path = "output"  # Define the output path
        self.tabs.setTabVisible(self.tabs.indexOf(self.scroll_areas["Processed Images"]), False)   
        registered_images = {}  # Initialize an empty dictionary to store registered images
        # Define paths for the R, G, and B channels
        
        red_channel_path = os.path.join(output_path, "R.png")
        green_channel_path = os.path.join(output_path, "G.png")
        blue_channel_path = os.path.join(output_path, "B.png")
        
        
    
        
        for image_type, image_path in self.selected_images.items():
            if image_type != "Vis" and image_path:
                # Perform image registration
                registered_image = perform_image_registration(self.selected_images["Vis"], image_path, output_path)

                # Construct the path for the registered image
                registered_image_path = os.path.join(output_path, f"registered_{image_type}.png")

                # Save the registered image
                cv2.imwrite(registered_image_path, registered_image)

                # Store the path of the registered image in the dictionary
                registered_images[image_type] = registered_image_path

                # Display the registered image (optional)
                self.display_image_in_tab(registered_image_path, self.image_label_align)    
        
        output_path = "output"
        if not os.path.exists(output_path):
                os.makedirs(output_path)
        # Save visible image to output folder
        vis_image_path = self.selected_images["Vis"]
        if vis_image_path:
            try:
                # Load the visible image
                vis_image = Image.open(vis_image_path)

                # Construct path for saving in the output folder
                output_vis_image_path = os.path.join(output_path, "Vis.png")

                # Save the visible image
                vis_image.save(output_vis_image_path)

                # Update the registered_images dictionary with the new path
                self.registered_images["Vis"] = output_vis_image_path  # Update for consistency

            except FileNotFoundError:
                raise FileNotFoundError(f"Visible image not found at: {vis_image_path}")


        if self.checkbox1.isChecked():
                # Register all images except for the visible image
            for image_type in self.selected_images:
                if image_type != "Vis" and self.selected_images[image_type]:
                        registered_image = perform_image_registration(vis_image_path, self.selected_images[image_type], output_path)
                        registered_image_path = os.path.join(output_path, f"registered_{image_type}.png")
                        cv2.imwrite(registered_image_path, registered_image)
                        self.registered_images[image_type] = registered_image_path
                        if registered_image_path:
                            self.display_image_in_tab(registered_image_path, self.image_label_align)
                            



        if self.checkbox2.isChecked():
                if self.checkbox1.isChecked():
                    viil_image_path = self.registered_images["VIIL"]
                else:
                    viil_image_path = self.selected_images["VIIL"]
                    
                output_path_irfc, error_message = run_IRFC(vis_image_path, viil_image_path, output_path)
                if output_path_irfc:
                    self.display_image_in_tab(output_path_irfc, self.image_label_irfc)
                else:
                    print(f"Error during IRFC: {error_message}")
                    
        if self.checkbox3.isChecked():
                if self.checkbox1.isChecked():
                    irr_image_path = self.registered_images["IRR"]
                else:
                    irr_image_path = self.selected_images["IRR"]
                    
                output_path_irfcirr, error_message = run_IRFC(vis_image_path, irr_image_path, output_path)
                if output_path_irfcirr:
                    self.display_image_in_tab(output_path_irfcirr, self.image_label_irfcirr)
                else:
                    print(f"Error during IRFC: {error_message}")
                    
        if self.checkbox4.isChecked():
                if self.checkbox1.isChecked():
                    uvr_image_path = self.registered_images["UVR"]
                else:
                    uvr_image_path = self.selected_images["UVR"]
                    
                output_path_uvr, error_message = run_UVFC(vis_image_path, uvr_image_path, output_path)
                if output_path_uvr:
                    self.display_image_in_tab(output_path_uvr, self.image_label_uvr)
                else:
                    print(f"Error during IRFC: {error_message}")
        
        
        
        if self.checkbox6.isChecked():
            vis_image_path = output_vis_image_path  # Ensure this is defined or passed correctly
            try:
                create_multiband_tiff(output_vis_image_path, output_path,)
                combine_multiband_tiff(registered_images, output_path, red_channel_path, green_channel_path, blue_channel_path)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to separate RGB channels: {str(e)}")


        
        self.update_tab_visibility()

    def reorder_thumbnails(self):
        for i in reversed(range(self.thumbnail_container.count())):
            widget = self.thumbnail_container.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        for image_type in self.image_order:
            file_path = self.selected_images[image_type]
            if file_path:
                self.add_thumbnail(image_type, file_path)

    def display_image_in_tab(self, image_path, label):
        image = QImage(image_path)
        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)
        label.setAlignment(Qt.AlignCenter)
        label.update()

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
            # Reset all checkboxes to unchecked
            self.checkbox1.setChecked(False)
            self.checkbox2.setChecked(False)
            self.checkbox3.setChecked(False)
            self.checkbox4.setChecked(False)
            self.checkbox5.setChecked(False)

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

        # Add thumbnails for each image type
        for image_type in self.image_order:
            file_path = self.selected_images[image_type]
            if file_path:
                self.add_thumbnail(image_type, file_path)

def main():
    app = QApplication([])
    window = ImageApp()
    window.show()
    app.exec_()

if __name__ == "__main__":
    main()
