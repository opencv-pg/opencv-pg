import logging

from qtpy import QtWidgets

from opencv_pg.models.transform_windows import collect_builtin_transforms
from opencv_pg.models.transform_list_model import TransformsModel

log = logging.getLogger(__name__)


class TransformList(QtWidgets.QTabWidget):
    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent=parent, *args, **kwargs)

        self.builtin_list = QtWidgets.QListView()
        self.custom_list = QtWidgets.QListView()
        self._load_builtins()

        self.addTab(self.builtin_list, "Built Ins")
        self.addTab(self.custom_list, "Custom")

    def _load_builtins(self):
        # TODO: Load the builtin list of transforms
        items = collect_builtin_transforms()
        model = TransformsModel(items)
        self.builtin_list.setModel(model)
