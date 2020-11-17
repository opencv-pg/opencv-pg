import logging

import numpy as np

from qtpy import QtCore

log = logging.getLogger(__name__)


class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, data):
        """Represents a 2d matrix

        Core from: https://www.learnpyqt.com/courses/model-views/qtableview-modelviews-numpy-pandas/
        """
        super().__init__()
        self._data = data
        self.editable = True

    def data(self, index, role):
        if role == QtCore.Qt.DisplayRole or role == QtCore.Qt.EditRole:
            return str(self._data[index.row()][index.column()])
        elif role == QtCore.Qt.TextAlignmentRole:
            return QtCore.Qt.AlignCenter

    @property
    def editable(self):
        return self._editable

    @editable.setter
    def editable(self, value):
        """Emit dataChanged when we change editbale to refresh any views"""
        index = QtCore.QModelIndex()
        self._editable = value
        self.dataChanged.emit(index, index)

    def flags(self, index):
        """Return Editable flags based on self.editable"""
        if self.editable:
            return QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled
        return QtCore.Qt.NoItemFlags

    def setData(self, index, value, role):
        if role == QtCore.Qt.EditRole:
            row = index.row()
            col = index.column()
            try:
                self._data[row][col] = float(value)
            except ValueError:
                log.debug("Unable to convert %s to float", value)
                return False

            self.dataChanged.emit(index, index)
            return True
        return False

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def resize(self, rows_in, cols_in, default=0):
        """Sets model shape to rows_in, cols_in. New cells = default"""
        self.beginResetModel()
        rows, cols = self._data.shape
        new_data = np.ones((rows_in, cols_in)) * default
        n_rows = rows_in if rows_in < rows else rows
        n_cols = cols_in if cols_in < cols else cols
        new_data[:n_rows, :n_cols] = self._data[:n_rows, :n_cols]
        self._data = new_data
        self.endResetModel()

    def set_internal_model_data(self, data_in):
        """Sets the models data to the incoming data"""
        self.beginResetModel()
        self._data = np.copy(data_in)
        self.endResetModel()

    def insertRow(self, row, index=QtCore.QModelIndex()):
        self.beginInsertRows(index, row, row)
        self._data = np.insert(
            self._data, obj=row, values=np.ones(self._data.shape[1]), axis=0
        )
        self.endInsertRows()
        return True

    def removeRow(self, row, index=QtCore.QModelIndex()):
        self.beginRemoveRows(index, row, row)
        self._data = np.delete(self._data, obj=row, axis=0)
        self.endRemoveRows()
        return True

    def insertColumn(self, col, index=QtCore.QModelIndex()):
        self.beginInsertColumns(index, col, col)
        self._data = np.insert(
            self._data, obj=col, values=np.ones(self._data.shape[0]), axis=1
        )
        self.endInsertColumns()
        return True

    def removeColumn(self, col, index=QtCore.QModelIndex()):
        self.beginRemoveColumns(index, col, col)
        self._data = np.delete(self._data, obj=col, axis=1)
        self.endRemoveColumns()
        return True
