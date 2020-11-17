from pathlib import Path

from qtpy.QtCore import Slot, QUrl
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import QMainWindow, QAction
from qtpy.QtWebEngineWidgets import QWebEngineView

from opencv_pg.docs import doc_writer


# Make sure rendered_docs folder exists
doc_writer._create_rendered_docs()


class MyWebView(QWebEngineView):
    def contextMenuEvent(self, event):
        self.parent().reload()
        super().contextMenuEvent(event)


class DocWindow(QMainWindow):
    def __init__(self, template_name):
        super().__init__()
        self.setWindowTitle("OpenCV DocViewer")
        self._setup_window_dims()
        self._add_statusbar(template_name)
        self._add_menus()
        self._load_webview(template_name)
        self._template_name = template_name

    def _load_webview(self, template_name):
        """Return a WebView widget"""
        view = MyWebView(self)
        doc_writer.render_local_doc(doc_writer.RENDERED_DIR, template_name)
        doc_name = doc_writer.RENDERED_DIR.joinpath(template_name)
        url = QUrl.fromLocalFile(str(doc_name))
        view.load(url)
        self.setCentralWidget(view)
        view.show()

    def reload(self):
        print("RELOAD")
        doc_writer.render_local_doc(doc_writer.RENDERED_DIR, self._template_name)
        doc_writer.RENDERED_DIR.joinpath(self._template_name)

    def _setup_window_dims(self):
        """Setup default window dimensions"""
        # TODO: Figure out how to get screen geometry
        # screen = QScreen()
        # geometry = screen.availableGeometry()
        width = 800
        height = 600
        self.resize(width, height)

    def _add_statusbar(self, fname):
        """Create and add a status bar"""
        self.status = self.statusBar()
        self.status.showMessage(fname)

    def _add_menus(self):
        """Create and add menubars"""
        self.menu = self.menuBar()
        self.menu.setNativeMenuBar(False)
        self.file_menu = self.menu.addMenu("File")

        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)

        self.file_menu.addAction(exit_action)
