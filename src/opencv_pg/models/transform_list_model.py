import logging

from qtpy import QtCore

log = logging.getLogger(__name__)


class TransformsModel(QtCore.QAbstractListModel):
    def __init__(self, items):
        self.items = items
        super().__init__(parent=None)

    def rowCount(self, parent):
        return len(self.items)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        row = index.row()

        if not index.isValid():
            return None

        if 0 > row >= len(self.data):
            return None

        if role == QtCore.Qt.DisplayRole:
            return self.items[index.row()].__name__

        return None
