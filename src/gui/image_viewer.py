import os
import subprocess
from PyQt5.QtWidgets import (QMainWindow, QAction, QFileDialog, QLabel, QScrollArea, QMessageBox,
                             QWidget, QHBoxLayout, QVBoxLayout, QFrame, QPushButton, QApplication, 
                            QCheckBox, QSizePolicy, QTabWidget, QStyle, QAbstractSlider, QSlider, QGridLayout, QLineEdit, QFileDialog )
from PyQt5.QtGui import QPixmap, QImage, QIcon, QTransform, QPainter, QColor
from PyQt5.QtCore import Qt, QPoint, QPointF, QRectF, QSizeF
import cv2
import numpy as np
import PIL
from PIL import Image
import shutil

# Import the image registration function
from utils.image_registration import perform_image_registration

from utils.irfcirr import run_IRFC_irr
from utils.uvfc import run_UVFC
from gui import image_viewer
from utils.create_mbtiff import create_multiband_tiff
from utils.process_images import process_images
from utils.histogram import calculate_histogram
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure





class ImageApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.title = "Multiband Image Processing"
        self.setWindowIcon(QIcon('icon.ico'))
        self.setGeometry(100, 100, 1500, 800)
        self.scroll_areas = {}  # Store scroll areas
        spectral_data = {}
        
        

        self.pixmap = None

        self.selected_images = {"Vis": None, "IRR": None, "VIIL": None, "UVR": None, "UVF": None}
        self.registered_images = {"Vis": None, "Aligned Vis": None, "Aligned IRR": None, "Aligned VIIL": None, "Aligned UVR": None, "Aligned UVF": None}

        self.image_order = ["UVR", "UVF", "Vis",  "IRR", "VIIL", "Aligned UVR", "Aligned UVF", "Aligned IRR", "Aligned VIIL", "Processed Images"]


        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.title)
        self.spectral_data = {}        
        
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
        button6 = QPushButton("660nm")
        button7 = QPushButton("735nm")

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
        button6.setStyleSheet("border-radius: 5px; background-color: white; padding: 5px;")
        button7.setStyleSheet("border-radius: 5px; background-color: white; padding: 5px;")
        

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
        self.tabs = QTabWidget()
        self.image_label_align = ZoomableLabel()
        self.image_label_irfc = ZoomableLabel()
        self.image_label_irfcirr = ZoomableLabel()
        self.image_label_uvfc = ZoomableLabel()
        self.image_label_processed = ZoomableLabel()
        self.image_label_multiband = ZoomableLabel()
        self.image_label_vis = ZoomableLabel()
        self.image_label_uvr = ZoomableLabel()
        self.image_label_irr = ZoomableLabel()
        self.image_label_uvf = ZoomableLabel()
        self.image_label_viil = ZoomableLabel()
        self.image_label_Reguvr = ZoomableLabel()
        self.image_label_Regirr = ZoomableLabel()
        self.image_label_Reguvf = ZoomableLabel()
        self.image_label_Regviil = ZoomableLabel()
        

        # Apply the same stylesheet settings to each ZoomableLabel
        for label in [self.image_label_align, self.image_label_irfc, self.image_label_irfcirr, self.image_label_uvfc,
                    self.image_label_processed, self.image_label_multiband, self.image_label_vis, self.image_label_uvr,
                    self.image_label_irr, self.image_label_uvf, self.image_label_viil, self.image_label_Reguvr,
                    self.image_label_Regirr, self.image_label_Reguvf, self.image_label_Regviil]:
            label.setStyleSheet("background-color: white; border-top-left-radius: 0px;")

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

         # Initialize tabs and set initial order
        self.init_tab(self.image_label_align, "Align")
        self.init_tab(self.image_label_vis, "Visible")
        self.init_tab(self.image_label_Reguvf, "Aligned UVF")
        self.init_tab(self.image_label_Reguvr, "Aligned UVR")
        self.init_tab(self.image_label_Regirr, "Aligned IRR")
        self.init_tab(self.image_label_Regviil, "Aligned VIIL")
        self.init_tab(self.image_label_irfc, "IRFC (VIIL)")
        self.init_tab(self.image_label_irfcirr, "IRFC (IRR)")
        self.init_tab(self.image_label_uvfc, "UVFC")
        self.init_tab(self.image_label_multiband, ("Multiband"))
        self.init_tab(self.image_label_processed, "Processed Images")
        self.init_tab(self.image_label_uvf, "UVF")
        self.init_tab(self.image_label_uvr, "UVR")
        self.init_tab(self.image_label_irr, "IRR")
        self.init_tab(self.image_label_viil, "VIIL")

        bottom_row = QFrame()
        bottom_row.setFrameShape(QFrame.StyledPanel)
        bottom_row_layout = QHBoxLayout(bottom_row)
        bottom_row.setStyleSheet("border: 0px; border-radius: 20px")
        
        # Additional Column (Newly Added)
        additional_column = QWidget()
        additional_column_layout = QVBoxLayout(additional_column)
        additional_column.setMaximumWidth(200)
        additional_column_layout.setAlignment(Qt.AlignCenter)
        additional_column.setStyleSheet("background-color: lightblue;")  # Adjust styling as needed

        # Example: Adding widgets to the additional column
        label = QLabel("Data")
        
        label.setAlignment(Qt.AlignCenter)
        labelSliders = QLabel("Edit")
        labelSliders.setAlignment(Qt.AlignCenter)
        label2 = QLabel("Edit Functions")
        label2.setAlignment(Qt.AlignCenter)
        

        histogram = QFrame()  # Example histogram widget
        histogram_layout = QVBoxLayout(histogram)
        histogram_layout.addWidget(label)
        histogram.setStyleSheet("background-color: white; border-radius: 5px;")  # Adjust styling as needed
        histogram.setFixedSize(150, 150)  # Adjust the size as needed

        
        
        
      
        # Layout setup
        layout = QVBoxLayout()  # Example layout
        slider_group = CustomSlider()

        

        #additional_column_layout.addWidget(label)
        #additional_column_layout.addWidget(histogram)
        #additional_column_layout.addWidget(labelSliders)
        #additional_column_layout.addWidget(slider_group.get_slider1())
        additional_column_layout.addWidget(label2)
        
        
        
        additional_column_layout.addStretch(1) 

        # Placeholder for future histograms or other functionalities

        bottom_row_layout.addWidget(additional_column)
        bottom_row_layout.addWidget(self.tabs)
        
        
        # Checkbox Column
        checkbox_column = QWidget()
        checkbox_column_layout = QVBoxLayout(checkbox_column)
        checkbox_column.setStyleSheet("background-color: lightblue")  # Adjust styling as needed
        
        label = QLabel("Processing Options")
        checkbox_column_layout.addWidget(label)
        
        
        self.checkbox1 = QCheckBox("Align")
        self.checkbox2 = QCheckBox("IRFC (VIIL)")
        self.checkbox3 = QCheckBox("IRFC (IRR)")
        self.checkbox4 = QCheckBox("UVFC")
        self.checkbox5 = QCheckBox("MBRS")
        self.checkbox6 = QCheckBox("Multiband TIFF")
        

        checkbox_column_layout.addWidget(self.checkbox1)
        checkbox_column_layout.addWidget(self.checkbox2)
        checkbox_column_layout.addWidget(self.checkbox3)
        checkbox_column_layout.addWidget(self.checkbox4)
        #checkbox_column_layout.addWidget(self.checkbox5)
        checkbox_column_layout.addWidget(self.checkbox6)
        

        process_button = QPushButton("Process")
        process_button.setStyleSheet("border-radius: 5px; background-color: white; padding: 10px;")
        process_button.clicked.connect(lambda: self.process_images())
        checkbox_column_layout.addWidget(process_button)
        checkbox_column_layout.addStretch(1)
        
        bottom_row_layout.addWidget(checkbox_column)
        
        main_layout.addWidget(bottom_row)
        
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        edit_menu = menubar.addMenu('Edit')
        # Create clear action
        clear_action = QAction('Clear', self)
        clear_action.triggered.connect(self.clear_images)
        exit_action = QAction('Exit', self)
        save_action = QAction('Save All', self)
        save_action.triggered.connect(self.save_all_images)
        file_menu.addAction(clear_action)
        file_menu.addAction(exit_action)
        file_menu.addAction(save_action)
   
        
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
        selected_images = self.selected_images
        checkboxes = {
            'checkbox1': self.checkbox1.isChecked(),
            'checkbox2': self.checkbox2.isChecked(),
            'checkbox3': self.checkbox3.isChecked(),
            'checkbox4': self.checkbox4.isChecked(),
            'checkbox6': self.checkbox6.isChecked()
        }
        registered_images = self.registered_images

        def display_image_callback(image_path, label_name):
            self.display_image_in_tab(image_path, getattr(self, label_name))

        process_images(
            selected_images,
            checkboxes,
            registered_images,
            display_image_callback,
            self.update_tab_visibility,
            
        )

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
    
    def display_spectral_data(self, spectral_data, pixel_x=None, pixel_y=None):
        """Displays a histogram of spectral data for a selected pixel or the average of the image using Matplotlib."""

        if pixel_x is not None and pixel_y is not None:
            # Display spectral profile for selected pixel
            band_values = spectral_data.get((pixel_x, pixel_y), [])
            title = f"Pixel ({pixel_x}, {pixel_y})"
        else:
            # Display average spectral profile of the entire image
            all_values = [val for pixel_values in spectral_data.values() for val in pixel_values]
            band_values = [np.mean(all_values[i::7]) for i in range(7)]  # Assuming 7 bands
            title = "Average Spectral Profile"

        # Create a Matplotlib figure and axes
        fig = Figure(figsize=(5, 4))  # Adjust size as needed
        ax = fig.add_subplot(111)

        # Plot the histogram
        ax.bar(range(len(band_values)), band_values, color='skyblue')
        ax.set_title(title)
        ax.set_xlabel('Band')
        ax.set_ylabel('Intensity')
        ax.set_xticks(range(len(band_values)))

        # Embed the plot in a PyQt5 widget
        canvas = FigureCanvas(fig)

        # Clear existing widgets in the histogram layout
        for i in reversed(range(self.histogram_layout.count())):
            self.histogram_layout.itemAt(i).widget().setParent(None) 

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

class ZoomableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self.scale_factor = 1.0
        self.offset = QPointF(0, 0)
        self.dragging = False
        self.drag_start_pos = QPointF(0, 0)

    def set_initial_pixmap(self, pixmap):
        self._pixmap = pixmap
        self.scale_factor = min(self.width() / pixmap.width(), self.height() / pixmap.height())
        self.offset = QPointF(0, 0)
        self.update_image()

    def update_image(self):
        if self._pixmap:
            transform = QTransform()
            transform.scale(self.scale_factor, self.scale_factor)
            scaled_pixmap = self._pixmap.transformed(transform, Qt.SmoothTransformation)

            display_pixmap = QPixmap(self.size())
            display_pixmap.fill(Qt.transparent)  # Start with a transparent pixmap

            painter = QPainter(display_pixmap)
            painter.drawPixmap(self.offset.toPoint(), scaled_pixmap)
            painter.end()

            super().setPixmap(display_pixmap)

    def wheelEvent(self, event):
        old_scale_factor = self.scale_factor

        if event.angleDelta().y() > 0:
            self.scale_factor *= 1.1
        else:
            self.scale_factor /= 1.1

        mouse_position = event.pos()
        offset_factor = (self.scale_factor / old_scale_factor) - 1
        self.offset -= QPointF(mouse_position.x() * offset_factor, mouse_position.y() * offset_factor)
        self.constrain_offset()
        self.update_image()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.drag_start_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self.dragging:
            delta = event.pos() - self.drag_start_pos
            self.drag_start_pos = event.pos()
            self.offset += QPointF(delta)
            self.constrain_offset()
            self.update_image()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
            self.setCursor(Qt.ArrowCursor)

    def constrain_offset(self):
        if self._pixmap:
            transform = QTransform()
            transform.scale(self.scale_factor, self.scale_factor)
            scaled_pixmap = self._pixmap.transformed(transform, Qt.SmoothTransformation)
            scaled_pixmap_rect = QRectF(self.offset, QSizeF(scaled_pixmap.size()))

            label_rect = self.rect()

            # Centering logic when zooming out
            if scaled_pixmap_rect.width() < label_rect.width():
                self.offset.setX((label_rect.width() - scaled_pixmap.width()) / 2)
            else:
                if scaled_pixmap_rect.left() > label_rect.left():
                    self.offset.setX(0)
                elif scaled_pixmap_rect.right() < label_rect.right():
                    self.offset.setX(label_rect.width() - scaled_pixmap.width())

            if scaled_pixmap_rect.height() < label_rect.height():
                self.offset.setY((label_rect.height() - scaled_pixmap.height()) / 2)
            else:
                if scaled_pixmap_rect.top() > label_rect.top():
                    self.offset.setY(0)
                elif scaled_pixmap_rect.bottom() < label_rect.bottom():
                    self.offset.setY(label_rect.height() - scaled_pixmap.height())

            self.update_image()

class MatplotlibWidget(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure()
        self.axes = fig.add_subplot(111)
        super(MatplotlibWidget, self).__init__(fig)
        self.setParent(parent)
        
    def plot_spectral_data(self, spectral_data):
        self.axes.clear()
        wavelengths = list(spectral_data.keys())
        intensities = list(spectral_data.values())
        self.axes.plot(wavelengths, intensities)
        self.axes.set_xlabel('Pixel Position')
        self.axes.set_ylabel('Intensity')
        self.axes.set_title('Spectral Data')
        self.draw()
        
class CustomSlider:
    def __init__(self):
        self.slider1 = QSlider(Qt.Horizontal)
        self.slider2 = QSlider(Qt.Horizontal)
        # ... other slider initialization
        for slider in [self.slider1, self.slider2]:
            slider.setMinimum(0)
            slider.setMaximumWidth(150)
            slider.setMaximum(10)

    def get_slider1(self):
        return self.slider1

    def get_slider2(self):
        return self.slider2
    
def display_spectral_data():
    wavelength = np.linspace(400, 700, 300)
    intensity = np.random.random(300)
    return {'wavelength': wavelength, 'intensity': intensity}

def main():
    app = QApplication([])
    app.setWindowIcon(QIcon('icon.ico'))  # Set the taskbar icon
    window = ImageApp()
    window.show()
    app.exec_()

if __name__ == "__main__":
    main()
