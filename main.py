# main.py

import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QPoint, QSize
from PyQt5.QtGui import QImage
import os


# Ensure the `src` directory is in the PYTHONPATH
sys.path.append(os.path.dirname(__file__))

from gui.image_viewer import ImageApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = ImageApp()
    main_window.show()
    sys.exit(app.exec_())
