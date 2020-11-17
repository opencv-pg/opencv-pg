from qtpy import QtWidgets, QtCore


class VerticalScrollArea(QtWidgets.QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setWidgetResizable(True)
