# main.py

import sys
from PyQt5.QtWidgets import QApplication
from gui.image_viewer import ImageApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = ImageApp()
    main_window.show()
    sys.exit(app.exec_())
