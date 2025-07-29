import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget
from ui.home import HomeWidget
from ui.manage_tools import ManageToolsWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("stEVE Guidewire Recommender")

        tabs = QTabWidget()
        tabs.addTab(HomeWidget(), "Home")
        tabs.addTab(ManageToolsWidget(), "Manage Tools")
        self.setCentralWidget(tabs)

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
