import logging
import numpy as np

from qtpy import QtWidgets, QtCore, QtGui
from opencv_pg.models import cv2_constants as cvc

import cv2

log = logging.getLogger(__name__)


class CenteredDelegate(QtWidgets.QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        widget = super().createEditor(parent, option, index)
        widget.setAlignment(QtCore.Qt.AlignCenter)
        return widget


class TableArray(QtWidgets.QTableView):
    active_cell_changed = QtCore.Signal(int, int)

    def __init__(self, parent=None, ctx_menu_label=None):
        super().__init__(parent=parent)
        self._ctx_menu_label = ctx_menu_label
        # Make the cells fit the alloted table space
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        self.setItemDelegate(CenteredDelegate())

    def event(self, event):
        """Enable context menu even if widget is disabled"""
        if event.type() == QtCore.QEvent.MouseButtonRelease:
            if event.button() == QtCore.Qt.RightButton:
                self.contextMenuEvent(event)
                event.accept()
                return True
        return super().event(event)

    def contextMenuEvent(self, event):
        """Conditionally display a right click Context Menu"""
        if not self._ctx_menu_label:
            return super().contextMenuEvent(event)

        menu = QtWidgets.QMenu(self)
        anchor_action = menu.addAction(self._ctx_menu_label)

        pos = event.pos()
        action = menu.exec_(self.mapToGlobal(pos))
        row = self.rowAt(pos.y())
        col = self.columnAt(pos.x())

        if action == anchor_action:
            self.active_cell_changed.emit(col, row)


class ArraySize(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(int, int)

    def __init__(self, prefix="Size", parent=None, min_val=1, max_val=100, width=35):
        """Widget to define two values like: (___, ___)

        Args:
            prefix (str, option): Text in front of first paren. Defaults to 'Size'
            parent (Widget, optional): Parent widget. Defaults to None.
            min_val (int, optional): Min value. Defaults to 1.
            max_val (int, optional): Max value. Defaults to 100.
            width (int, optional): Width of QLineEdit in pixels. Defaults to 30.
        """
        super().__init__(parent=parent)
        hbox = QtWidgets.QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        self.min_val = min_val
        self.max_val = max_val

        # Labels
        if prefix:
            prefix += ": "
        begin_paren = QtWidgets.QLabel(f"{prefix}(")
        comma = QtWidgets.QLabel(",")
        end_paren = QtWidgets.QLabel(")")

        # LineEdits
        self.row_input = self._get_line_edit(width=width)
        self.col_input = self._get_line_edit(width=width)

        # Add widgets
        hbox.addWidget(begin_paren)
        hbox.addWidget(self.row_input)
        hbox.addWidget(comma)
        hbox.addWidget(self.col_input)
        hbox.addWidget(end_paren)
        hbox.addStretch()

        # Make sure line edits are left-aligned
        hbox.setAlignment(self.row_input, QtCore.Qt.AlignLeft)
        hbox.setAlignment(self.col_input, QtCore.Qt.AlignLeft)
        self.setLayout(hbox)

        self.row_input.editingFinished.connect(self._handle_value_changed)
        self.col_input.editingFinished.connect(self._handle_value_changed)

    def _get_line_edit(self, width):
        """Return a fixed width QLineEdit with integer validator"""
        line_edit = QtWidgets.QLineEdit(alignment=QtCore.Qt.AlignCenter)
        line_edit.setFixedWidth(width)
        validator = QtGui.QIntValidator(self.min_val, self.max_val)
        line_edit.setValidator(validator)
        return line_edit

    def set_text(self, rows, cols):
        self.row_input.setText(str(rows))
        self.col_input.setText(str(cols))

    @QtCore.Slot()
    def _handle_value_changed(self):
        row_text = self.row_input.text()
        col_text = self.col_input.text()

        # Don't want to emit changed if either field is empty or < min
        if (
            not row_text
            or not col_text
            or int(row_text) < self.min_val
            or int(col_text) < self.min_val
        ):
            return

        self.valueChanged.emit(int(row_text), int(col_text))


class Array1DSize(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(int)

    def __init__(self, prefix="Size", parent=None, min_val=1, max_val=100, width=35):
        """Widget to define one value like: (___)

        Args:
            prefix (str, option): Text in front of first paren. Defaults to 'Size'
            parent (Widget, optional): Parent widget. Defaults to None.
            min_val (int, optional): Min value. Defaults to 1.
            max_val (int, optional): Max value. Defaults to 100.
            width (int, optional): Width of QLineEdit in pixels. Defaults to 30.
        """
        super().__init__(parent=parent)
        hbox = QtWidgets.QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        self.min_val = min_val
        self.max_val = max_val

        # Labels
        if prefix:
            prefix += ": "
        begin_paren = QtWidgets.QLabel(f"{prefix}(")
        end_paren = QtWidgets.QLabel(")")

        # LineEdits
        self.col_input = self._get_line_edit(width=width)

        # Add widgets
        hbox.addWidget(begin_paren)
        hbox.addWidget(self.col_input)
        hbox.addWidget(end_paren)
        hbox.addStretch()

        # Make sure line edits are left-aligned
        hbox.setAlignment(self.col_input, QtCore.Qt.AlignLeft)
        self.setLayout(hbox)

        self.col_input.editingFinished.connect(self._handle_value_changed)

    def _get_line_edit(self, width):
        """Return a fixed width QLineEdit with integer validator"""
        line_edit = QtWidgets.QLineEdit(alignment=QtCore.Qt.AlignCenter)
        line_edit.setFixedWidth(width)
        validator = QtGui.QIntValidator(self.min_val, self.max_val)
        line_edit.setValidator(validator)
        return line_edit

    def set_text(self, cols):
        self.col_input.setText(str(cols))

    @QtCore.Slot()
    def _handle_value_changed(self):
        col_text = self.col_input.text()

        # Don't want to emit changed if either field is empty or < min
        if not col_text or int(col_text) < self.min_val:
            return

        self.valueChanged.emit(int(col_text))


class BaseArrayEditor(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(np.ndarray)
    anchorChanged = QtCore.Signal(int, int)
    anchor1DChanged = QtCore.Signal(int)

    def __init__(self, model, use_anchor=True, parent=None, dims=2):
        super().__init__(parent=parent)
        self._use_anchor = use_anchor
        rows, cols = model._data.shape
        self.vbox = QtWidgets.QVBoxLayout()
        self.vbox.setSpacing(5)
        self.vbox.setContentsMargins(0, 5, 0, 0)
        self.dims = dims

        if dims == 1:
            self.array_size = Array1DSize(parent=self)
            self.array_size.set_text(cols)
        else:
            self.array_size = ArraySize(parent=self)
            self.array_size.set_text(rows, cols)

        menu_label = None
        if use_anchor:
            menu_label = "Set Anchor"
        self.array = TableArray(parent=self, ctx_menu_label=menu_label)
        self.array.setModel(model)

        self.vbox.addWidget(self.array_size)
        self.vbox.addWidget(self.array)
        if use_anchor:
            if self.dims == 1:
                self._anchor = -1
                self.anchor = QtWidgets.QLabel("Anchor: (-1)")
            else:
                self._anchor = (-1, -1)
                self.anchor = QtWidgets.QLabel("Anchor: (-1, -1)")
            self.vbox.addWidget(self.anchor)
        self.setLayout(self.vbox)

    def _connect_signals(self, model):
        """Make internal signal connections between widgets"""
        if self.dims == 1:
            self.array_size.valueChanged.connect(self._handle_1Dsize_change)
        else:
            self.array_size.valueChanged.connect(self._handle_size_change)
        model.dataChanged.connect(self._handle_data_changed)
        if self._use_anchor:
            if self.dims == 1:
                self.array.active_cell_changed.connect(self._handle_anchor1D_change)
            else:
                self.array.active_cell_changed.connect(self._handle_anchor_change)

    @QtCore.Slot(int, int)
    def _handle_anchor_change(self, row, col):
        self.anchor.setText(f"Anchor: ({row}, {col})")
        self._anchor = (row, col)
        self.anchorChanged.emit(row, col)

    @QtCore.Slot(int)
    def _handle_anchor1D_change(self, col):
        self.anchor.setText(f"Anchor: ({col})")
        self._anchor = col
        self.anchor1DChanged.emit(col)

    @QtCore.Slot(QtCore.QModelIndex, QtCore.QModelIndex)
    def _handle_data_changed(self, idx1, idx2):
        model = self.array.model()
        self.valueChanged.emit(model._data)

    @QtCore.Slot(int, int)
    def _handle_size_change(self, rows, cols):
        model = self.array.model()
        model.resize(rows, cols)
        if self._use_anchor:
            anchor_reset = self._check_and_reset_anchor(rows, cols)
            if anchor_reset:
                if self.dims == 1:
                    self.anchorChanged.emit(-1)
                else:
                    self.anchorChanged.emit(-1, -1)
        self.valueChanged.emit(model._data)

    @QtCore.Slot(int)
    def _handle_1Dsize_change(self, cols):
        model = self.array.model()
        model.resize(1, cols)
        if self._use_anchor:
            anchor_reset = self._check_and_reset_anchor(1, cols)
            if anchor_reset:
                self.anchorChanged.emit(-1)
        self.valueChanged.emit(model._data)

    def _check_and_reset_anchor(self, data_rows, data_cols):
        """If Anchor is OOB, reset it to -1, -1"""
        if self.dims == 1:
            col = self._anchor
            if (col + 1) > data_cols:
                log.debug("Anchor was %s, now OOB. Reset to -1", self._anchor)
                self._anchor = -1
                return True
        else:
            row, col = self._anchor
            if (row + 1) > data_rows or (col + 1) > data_cols:
                log.debug("Anchor was %s, now OOB. Reset to -1, -1", self._anchor)
                self._anchor = (-1, -1)
                return True
        return False


class ArrayEditor(BaseArrayEditor):
    def __init__(self, model, use_anchor=True, parent=None, dims=2):
        super().__init__(model, use_anchor=use_anchor, parent=parent, dims=dims)
        self._connect_signals(model)


class ArrayEditorWithStructElement(BaseArrayEditor):
    def __init__(self, model, use_anchor=True, parent=None):
        super().__init__(model, use_anchor=use_anchor, parent=parent)
        hbox = QtWidgets.QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)

        self.struct = QtWidgets.QComboBox(parent=self)
        for opt in list(cvc.MORPH_SHAPES.keys()):
            self.struct.addItem(opt)
        self.struct.addItem("Custom")

        label = QtWidgets.QLabel("Structuring Element:")
        hbox.addWidget(label)
        hbox.addWidget(self.struct)
        hbox.addStretch()

        container = QtWidgets.QWidget()
        container.setLayout(hbox)
        self.vbox.insertWidget(0, container)
        self._connect_signals(model)

    def _connect_signals(self, model):
        super()._connect_signals(model)
        self.array.model().editable = False
        self.struct.currentTextChanged.connect(self._handle_struct_change)

    @QtCore.Slot(str)
    def _handle_struct_change(self, text):
        if text == "Custom":
            self.array.model().editable = True
            return

        shape = cvc.MORPH_SHAPES[text]
        model = self.array.model()
        el = cv2.getStructuringElement(shape, model._data.shape)
        model.set_internal_model_data(el)
        self.array.model().editable = False
        self.valueChanged.emit(model._data)

    @QtCore.Slot(int, int)
    def _handle_size_change(self, rows, cols):
        model = self.array.model()
        shape = self.struct.currentText()

        # Use the morph shape if it's set
        if shape != "Custom":
            el = cv2.getStructuringElement(cvc.MORPH_SHAPES[shape], (rows, cols))
            model.set_internal_model_data(el)
            self.valueChanged.emit(model._data)
            return

        # Model is Custom so don't touch internal structure
        model.resize(rows, cols)
        if self._use_anchor:
            anchor_reset = self._check_and_reset_anchor(rows, cols)
            if anchor_reset:
                if self.dims == 1:
                    self.anchorChanged.emit(-1)
                else:
                    self.anchorChanged.emit(-1, -1)

        self.valueChanged.emit(model._data)
