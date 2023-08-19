"""Testing building some GUI components

This format is not going to be used. View model probably works better
"""
import logging

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from opencv_pg.models.window import Window

from .widgets.image_viewer import ImageViewer
from .widgets.vertical_scroll_area import VerticalScrollArea

log = logging.getLogger(__name__)


class PipeWindow(QtWidgets.QWidget):
    def __init__(self, window: Window, parent=None, show_info_widget=True):
        super().__init__(parent=parent)
        self.show_info_widget = show_info_widget
        self.window = window
        self.viewer = None
        layout = self._build_layout()
        self.setLayout(layout)
        self.window.image_updated.connect(self._handle_pipeline_completed)

    def set_splitter_pos(self, percent):
        """Set the splitter `percent` of the way down the window"""
        y = self.size().height()
        splitter = self.children()[1]
        new_pos = int(y * percent)
        splitter.moveSplitter(new_pos, 1)

    def resizeEvent(self, event):
        """Scale the image viewer scene with the window size change"""
        self.viewer.update_scale()
        super().resizeEvent(event)

    @QtCore.Slot(int, int)
    def handle_splitter_moved(self, pos, index):
        """Rescale viewer image when splitter changes"""
        self.viewer.update_scale()

    def update_image(self, img_in, viewer):
        """Update the `viewer` with `img_in`"""
        q_img = self._get_qimage(img_in)
        pixmap = QtGui.QPixmap.fromImage(q_img)
        viewer.setPixmap(pixmap)

    def _get_qimage(self, img):
        """Return a QImage appropriate for the image type

        We're going to do this the same way that cv2.imshow works:
        https://github.com/opencv/opencv/blob/bea2c7545243ba2dabce6badc94dd55894a8e5ca/modules/highgui/include/opencv2/highgui.hpp#L358-L366
        """
        imtype = img.dtype
        y, x = img.shape[0], img.shape[1]
        channels = 1 if len(img.shape) == 2 else img.shape[2]

        # Scale image from 0, 255
        # maybe dropping precision since using astype and not rounding first ...
        if imtype in (np.uint16, np.int32):
            img = (img / 256).astype(np.uint8)
        elif imtype in (np.float32, np.float64):
            img = (img * 255).astype(np.uint8)
        elif imtype != np.uint8:
            log.error(
                "'%s' is not a supported np.ndarray type. Supported types "
                "are: uint8, uint16, int32, float32, and float64.",
                imtype,
            )

        # Display as grayscale or BGR
        if len(img.shape) == 2:
            fmt = QtGui.QImage.Format_Grayscale8
        else:
            fmt = QtGui.QImage.Format_BGR888
        return QtGui.QImage(img.data, x, y, channels * x, fmt)

    @QtCore.Slot()
    def _handle_pipeline_completed(self):
        """Redraw the output image after a param change"""
        self.update_image(self.window.last_out, self.viewer)

    def _build_layout(self):
        """Build the Pipeline Window Layout

        - QVBoxLayout (top_layou)
          |- QSplitter (v_splitter)
            |- ImageViewer (self.viewer)
            |- QScrollArea (controls_scroll)
              |- QWidget (controls_widget)
                |- QVBoxLayout (transform_vlayout)
                    |- Widgets (via Param.get_widget)
                    |- ...
        """
        top_layout = QtWidgets.QVBoxLayout()
        v_splitter = QtWidgets.QSplitter(orientation=QtCore.Qt.Vertical)
        v_splitter.splitterMoved.connect(self.handle_splitter_moved)
        top_layout.addWidget(v_splitter)

        # Rendered image
        self.viewer = ImageViewer()

        # Build Transform groups and their controls
        controls_scroll = VerticalScrollArea(parent=v_splitter)
        controls_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        transform_vlayout = QtWidgets.QVBoxLayout()

        for transform in self.window.transforms:
            if self.show_info_widget:
                info_widget = transform.get_info_widget()
                if info_widget:
                    transform_vlayout.addWidget(info_widget)

            if not transform.params:
                continue

            # Build the groupbox with its transforms
            groupbox = WidgetGroup(transform.__class__.__name__)
            for param in transform.params:
                widget = param.get_widget()
                groupbox.add_widget(widget, param.label, param.help_text)
                widget.setParent(groupbox)
                if param.read_only:
                    param.set_enabled(False)

            transform.interconnect_widgets()
            groupbox.toggled.connect(transform.handle_enabled_changed)

            # Add groupbox to tranform layout and
            transform_vlayout.addWidget(groupbox)
        transform_vlayout.addStretch()

        controls_widget = QtWidgets.QWidget()
        controls_widget.setLayout(transform_vlayout)

        controls_scroll.setWidget(controls_widget)

        v_splitter.addWidget(self.viewer)
        v_splitter.addWidget(controls_scroll)

        # Let viewer have as much space as it can take
        v_splitter.setStretchFactor(0, 1)
        return top_layout


class WidgetGroup(QtWidgets.QGroupBox):
    def __init__(self, name, parent=None):
        super().__init__(name, parent=parent)
        self.form_layout = QtWidgets.QFormLayout()
        self.form_layout.setVerticalSpacing(5)
        self.form_layout.setHorizontalSpacing(10)
        self.form_layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)
        self.setLayout(self.form_layout)
        self.setCheckable(True)
        self.setChecked(True)

    def add_widget(self, widget, label, help_text=""):
        qlabel = QtWidgets.QLabel(f"{label}:")
        if help_text:
            qlabel.setToolTip(help_text)
        self.form_layout.addRow(qlabel, widget)

    def change_label(self, widget, label):
        """Change the label for the provided widget"""
        qlabel = self.form_layout.labelForField(widget)
        qlabel.setText(label)
