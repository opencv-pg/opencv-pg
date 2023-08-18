import logging

from qtpy import QtWidgets
from qtpy.QtCore import QModelIndex, Qt, QUrl, Slot
from qtpy.QtWebEngineWidgets import QWebEngineView

from opencv_pg.docs.doc_writer import RENDERED_DIR
from opencv_pg.models.pipeline import Pipeline
from opencv_pg.models.transform_windows import get_transform_window

from .pipeline_window import PipeWindow
from .widgets.transform_list import TransformList

log = logging.getLogger(__name__)


class Playground(QtWidgets.QSplitter):
    def __init__(
        self, img_path, no_docs, disable_info_widgets, parent=None, *args, **kwargs
    ):
        super().__init__(parent=parent, *args, **kwargs)
        self.img_path = str(img_path)
        self.show_docs = not no_docs
        self.show_info_widgets = not disable_info_widgets
        self.docview = None
        self.pipe_stack = None
        self.added_pipes = {}

        self.setOrientation(Qt.Orientation.Horizontal)
        self._build_layout()
        # TODO: Distribution based on screen size
        self.setSizes([0.1 * 800, 0.5 * 800, 0.4 * 800])

    def _build_layout(self):
        """Build the main layout"""
        tlist = TransformList()
        # TODO: Use some max based on largest object in list size
        tlist.setMinimumWidth(150)
        tlist.setMaximumWidth(200)
        self.addWidget(tlist)

        # Connect change handlers for the transform list
        if self.show_docs:
            tlist.builtin_list.selectionModel().currentChanged.connect(
                self._handle_changed
            )
        tlist.builtin_list.selectionModel().currentChanged.connect(
            self._reload_pipeline
        )

        # Document Viewer
        if self.show_docs:
            self.docview = QWebEngineView(parent=self)
            self.addWidget(self.docview)
            # NOTE: No idea why, but we get random segfaults if we don't first set/load
            # some kind of html before the signal handler takes over ... :shrug:
            self.docview.setHtml("")

        # PipeWindow
        self.pipe_stack = QtWidgets.QStackedWidget(parent=self)
        self.addWidget(self.pipe_stack)

    @Slot(QModelIndex, QModelIndex)
    def _reload_pipeline(self, current, previous):
        """Creates or Loads selected Transform into PipeWindow"""
        model = current.model()
        transform = model.items[current.row()]
        tname = transform.__name__

        # Add or select a PipeWindow from the stack
        if self.pipe_stack.currentIndex() == -1 or tname not in self.added_pipes:
            window = get_transform_window(transform, self.img_path)
            pipe = Pipeline(window)
            pipe_win = PipeWindow(
                window, parent=self, show_info_widget=self.show_info_widgets
            )
            img, _ = pipe.run_pipeline()
            pipe_win.update_image(img, pipe_win.viewer)
            self.pipe_stack.addWidget(pipe_win)
            self.added_pipes[tname] = self.pipe_stack.count() - 1
            self.pipe_stack.setCurrentIndex(self.added_pipes[tname])
        else:
            self.pipe_stack.setCurrentIndex(self.added_pipes[tname])

    @Slot(QModelIndex, QModelIndex)
    def _handle_changed(self, current, previous):
        """Reloads documentation for selected Transform"""
        model = current.model()
        transform = model.items[current.row()]
        doc_fname = RENDERED_DIR.joinpath(transform.get_doc_filename())
        url = QUrl.fromLocalFile(str(doc_fname))
        self.docview.load(url)
