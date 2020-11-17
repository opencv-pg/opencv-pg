from qtpy.QtGui import QKeySequence, QGuiApplication
from qtpy.QtWidgets import QMainWindow, QAction
from .views import playground


class MainWindow(QMainWindow):
    def __init__(self, img_path, no_docs):
        super().__init__()
        self.setWindowTitle("OpenCV PlayGround")
        self._setup_window_size(0.5)
        self._add_statusbar()
        self._add_menus()

        # TODO: set this with either playground or designer
        self.setCentralWidget(playground.Playground(img_path, no_docs))

    def _setup_window_size(self, fraction):
        """Setup default window dimensions"""
        screen = QGuiApplication.primaryScreen()
        geometry = screen.geometry()
        width = geometry.width() * fraction
        height = geometry.height() * fraction
        self.resize(width, height)

    def _add_statusbar(self):
        """Create and add a status bar"""
        self.status = self.statusBar()
        self.status.showMessage("Welcome to OpenCV Playground")

    def _add_menus(self):
        """Create and add menubars"""
        self.menu = self.menuBar()
        self.menu.setNativeMenuBar(False)
        self.file_menu = self.menu.addMenu("File")

        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)

        self.file_menu.addAction(exit_action)
