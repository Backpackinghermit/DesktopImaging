import os
import subprocess
from PyQt5.QtWidgets import (QMainWindow, QAction, QFileDialog, QLabel, QScrollArea, QMessageBox,
                             QWidget, QHBoxLayout, QVBoxLayout, QFrame, QPushButton, QApplication, 
                             QCheckBox, QSizePolicy, QTabWidget, QStyle )
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import numpy as np

# Import the image registration function
from utils.image_registration import perform_image_registration
from utils.IRFC import run_IRFC  # Import the IRFC function

class ImageApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.title = "Conservation Image Processing"
        self.setGeometry(100, 100, 1500, 800)

        self.pixmap = None

        self.selected_images = {"Vis": None, "IRR": None, "VIIL": None}
        self.registered_images = {"Vis": None, "IRR": None, "VIIL": None}

        self.image_order = ["Vis", "IRR", "VIIL"]

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

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        section1 = QVBoxLayout()
        button1 = QPushButton("Vis")
        button2 = QPushButton("IRR")
        button3 = QPushButton("VIIL")

        button1.clicked.connect(lambda: self.open_image_and_display_thumbnail("Vis"))
        button2.clicked.connect(lambda: self.open_image_and_display_thumbnail("IRR"))
        button3.clicked.connect(lambda: self.open_image_and_display_thumbnail("VIIL"))

        section1.addWidget(button1)
        section1.addWidget(button2)
        section1.addWidget(button3)
        button1.setStyleSheet("border-radius: 5px; background-color: white; padding: 5px;")
        button2.setStyleSheet("border-radius: 5px; background-color: white; padding: 5px;")
        button3.setStyleSheet("border-radius: 5px; background-color: white; padding: 5px;")
        

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

        top_row_widget = QWidget()
        top_row_widget.setLayout(top_row_layout)
        scroll_area.setWidget(top_row_widget)
        scroll_area.setMaximumHeight(200)
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
        self.image_label_other = QLabel()
        self.image_label_align.setStyleSheet("background-color: white; border-top-left-radius: 0px;")
        self.image_label_irfc.setStyleSheet("background-color: white; border-top-left-radius: 0px;")
        self.image_label_irfcirr.setStyleSheet("background-color: white; border-top-left-radius: 0px;")
        self.image_label_other.setStyleSheet("background-color: white; border-top-left-radius: 0px;")
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

        right_column_layout.addWidget(self.checkbox1)
        right_column_layout.addWidget(self.checkbox2)
        right_column_layout.addWidget(self.checkbox3)
        right_column_layout.addWidget(self.checkbox4)
        right_column_layout.addWidget(self.checkbox5)

        process_button = QPushButton("Process")
        process_button.setStyleSheet("border-radius: 5px; background-color: white;")
        process_button.clicked.connect(self.process_images)
        right_column_layout.addWidget(process_button)

        bottom_row_layout.addWidget(right_column)
        main_layout.addWidget(bottom_row)
        
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')

        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
    def init_tab(self, label, title):
        label.setAlignment(Qt.AlignCenter)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        scroll_area = QScrollArea()
        scroll_area.setWidget(label)
        scroll_area.setWidgetResizable(True)
        self.tabs.addTab(scroll_area, title)
        scroll_area.setStyleSheet(""""
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
        
        

    def open_image_and_display_thumbnail(self, image_type):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, f"Open {image_type} Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        if file_path:
            if self.selected_images[image_type]:
                self.selected_images[image_type] = None

                item = self.thumbnail_container.itemAt(0)
                if item:
                    item.widget().deleteLater()

            thumbnail_label = QLabel()
            pixmap = QPixmap(file_path)
            thumbnail_label.setPixmap(pixmap.scaledToHeight(130, Qt.SmoothTransformation))
            self.thumbnail_container.addWidget(thumbnail_label)
            self.selected_images[image_type] = file_path

            self.reorder_thumbnails()

    def reorder_thumbnails(self):
        for i in reversed(range(self.thumbnail_container.count())):
            widget = self.thumbnail_container.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        for image_type in self.image_order:
            file_path = self.selected_images[image_type]
            if file_path:
                thumbnail_label = QLabel()
                pixmap = QPixmap(file_path)
                thumbnail_label.setPixmap(pixmap.scaledToHeight(200, Qt.SmoothTransformation))
                self.thumbnail_container.addWidget(thumbnail_label)

    def process_images(self):
        irr_image_path = self.selected_images["IRR"]
        vis_image_path = self.selected_images["Vis"]
        viil_image_path = self.selected_images["VIIL"]
        
        if irr_image_path and vis_image_path:
            output_path = "output"
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            if self.checkbox1.isChecked():
                # Register all images except for the visible image
                for image_type in self.selected_images:
                    if image_type != "Vis" and self.selected_images[image_type]:
                        registered_image = perform_image_registration(vis_image_path, self.selected_images[image_type], output_path)
                        registered_image_path = os.path.join(output_path, f"registered_{image_type}.png")
                        cv2.imwrite(registered_image_path, registered_image)
                        self.registered_images[image_type] = registered_image_path

                        self.update_thumbnail(image_type, registered_image_path)

                self.reorder_thumbnails()

                # Display the aligned image if it exists
                if self.registered_images["IRR"]:
                    self.display_image_in_tab(self.registered_images["IRR"], self.image_label_align)

            if self.checkbox2.isChecked():
                # Use the registered VIIL image if alignment was performed
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
                # Use the registered IRR image if alignment was performed
                if self.checkbox1.isChecked():
                    irr_image_path = self.registered_images["IRR"]
                else:
                    irr_image_path = self.selected_images["IRR"]
                    
                output_path_irfcirr, error_message = run_IRFC(vis_image_path, irr_image_path, output_path)
                if output_path_irfcirr:
                    self.display_image_in_tab(output_path_irfcirr, self.image_label_irfcirr)
                else:
                    print(f"Error during IRFC: {error_message}")
        else:
            QMessageBox.warning(self, "Warning", "Please select processing methods")


    def update_thumbnail(self, image_type, image_path):
        for i in range(self.thumbnail_container.count()):
            item = self.thumbnail_container.itemAt(i)
            if item and item.widget():
                pixmap = item.widget().pixmap()
                if pixmap and not pixmap.isNull():  # Check if pixmap is not null
                    current_image_path = self.selected_images[image_type]
                    if current_image_path and current_image_path == image_path:
                        new_pixmap = QPixmap(image_path)
                        item.widget().setPixmap(new_pixmap.scaledToHeight(150, Qt.SmoothTransformation))
                        break


    def display_image_in_tab(self, image_path, label):
        image = QImage(image_path)
        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)



    def resizeEvent(self, event):
            super().resizeEvent(event)
            for i in range(self.tabs.count()):
                tab = self.tabs.widget(i)
                if tab:
                    image_label = tab.widget()
                    if image_label and image_label.pixmap():
                        scaled_pixmap = image_label.pixmap().scaled(image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        image_label.setPixmap(scaled_pixmap)

def main():
    app = QApplication([])
    window = ImageApp()
    window.show()
    app.exec_()

if __name__ == "__main__":
    main()
