import logging

from qtpy import QtCore, QtWidgets

log = logging.getLogger(__name__)


class ImageViewer(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, False)
        self.setDragMode(self.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(self.ViewportAnchor.AnchorViewCenter)
        self.setResizeAnchor(self.ViewportAnchor.AnchorViewCenter)

        self._scene = QtWidgets.QGraphicsScene(self)
        self._img = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._img)
        self.setScene(self._scene)

        self._base_zoom_fac = 0.002

        self._first_time_set = False
        self._current_rect = None

    def setPixmap(self, pixmap, save=True):
        self._img.setPixmap(pixmap)
        self._current_rect = self.get_visible_rect()

    def mouseReleaseEvent(self, event):
        """Store visible rect when done dragging with LMB (panning)"""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._current_rect = self.get_visible_rect()
        return super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        """Fit the image to the window the first time it's loaded"""
        if not self._first_time_set:
            self.reset_view()
            self._first_time_set = True
        super().resizeEvent(event)

    def wheelEvent(self, event):
        """Handle zooming in and out"""
        pixels = event.pixelDelta()

        # Qt Suggests preferring pixel based movement when available
        if not pixels.isNull():
            zoom = self._zoom_fac_by_pixel_change(pixels)
        else:
            degrees = event.angleDelta()
            zoom = self._zoom_fac_by_degree_change(degrees)

        self._handle_zoom(zoom)
        self._current_rect = self.get_visible_rect()
        event.accept()

    def _is_max_zoomed_out(self):
        """Return True if neither scrollbar is active, thus fully zoomed out"""
        scroll_h = self.horizontalScrollBar().isVisible()
        scroll_v = self.verticalScrollBar().isVisible()
        if not scroll_h and not scroll_v:
            return True
        return False

    def _zoom_fac_by_degree_change(self, delta):
        """Return zoom factor based on relative change of degrees of mousewheel

        Args:
            delta (QPoint): Relative amount rotated in 1/8ths of a degree.
                Value in delta.y()

        Returns:
            float: Scaling factor
        """
        return 1 + self._base_zoom_fac * delta.y() * 0.6

    def _zoom_fac_by_pixel_change(self, delta):
        """Return a zoom factor based on pixel change

        Args:
            delta (QPoint): Number of vertical pixels changed is delta.y()

        Returns:
            float: Scaling factor
        """
        return 1 + self._base_zoom_fac * delta.y()

    def _handle_zoom(self, zoom_fac):
        """Perform zooming in/out"""
        if zoom_fac > 1:
            self.scale(zoom_fac, zoom_fac)
        elif self._is_max_zoomed_out():
            return
        self.scale(zoom_fac, zoom_fac)

    def update_scale(self):
        """Scales the currently displayed image during viewport resize

        Called by parent widget's resizeEvent
        """
        if self._is_max_zoomed_out():
            self.reset_view()
        self.fitInView(self._current_rect, QtCore.Qt.KeepAspectRatio)

    def contextMenuEvent(self, event):
        """Add a reset view context menu"""
        menu = QtWidgets.QMenu(self)
        reset_action = menu.addAction("Reset View")
        pos = event.pos()
        action = menu.exec_(self.mapToGlobal(pos))
        if action == reset_action:
            self.reset_view()

    def reset_view(self):
        self.fitInView(self._img, QtCore.Qt.KeepAspectRatio)
        self._current_rect = self.get_visible_rect()

    def get_visible_rect(self):
        """Return the part of the scene that is visible in the viewport"""
        return self.mapToScene(self.viewport().rect()).boundingRect()
