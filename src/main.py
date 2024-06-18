# main.py

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QIcon
from gui.image_viewer import ImageApp
from gui.image_viewer import ImageApp, ZoomableLabel, CustomSlider

 



if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = ImageApp()
    main_window.show()
    sys.exit(app.exec_())
