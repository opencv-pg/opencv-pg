import logging
from typing import List, Union

from .window import Window
from .base_transform import BaseTransform

Windows = List[Window]
Transforms = List[BaseTransform]

log = logging.getLogger(__name__)


class Pipeline:
    """An image processing Pipeline"""

    def __init__(self, items: Union[Windows, Transforms, BaseTransform, Window]):
        self.windows = self._create_windows(items)
        self._init_pipeline()

    def run_pipeline(self, win_index: int = 0, transform_index: int = 0):
        """Run pipeline from Window ``win_index`` and Transform ``transform_index``"""
        img_out = None
        extra_out = None
        for window in self.windows[win_index:]:
            img_out, extra_out = window.draw(img_out, extra_out, transform_index)
            # Only want to start win_index at transform_index; every other at 0
            transform_index = 0
        return img_out, extra_out

    def get_transform(self, win_index: int, trans_index: int) -> BaseTransform:
        """Returns Transform at ``win_index``, ``trans_index``

        Args:
            win_index (int): Window Index
            trans_index (int): Transform Index within Window

        Returns:
            BaseTransform: Transform
        """
        return self.windows[win_index].transforms[trans_index]

    def _init_pipeline(self):
        """Sets up relationships within pipeline windows and transforms"""
        for w_idx, window in enumerate(self.windows):
            window.pipeline = self
            window.index = w_idx
            for t_idx, transform in enumerate(window.transforms):
                transform.window = window
                transform.index = t_idx

    def _create_windows(
        self, items: Union[Windows, Transforms, BaseTransform, Window]
    ) -> Windows:
        """Return appropriate Window(s) based on type of items"""
        if isinstance(items, BaseTransform):
            return [Window([items])]

        if isinstance(items, Window):
            return [items]

        if isinstance(items, list):
            # Ensure all items are the same type
            self._all_same_class(items)
            if isinstance(items[0], Window):
                return items
            else:
                return [Window(items)]

        raise TypeError(f"Unsupported type: {items.__class__}")

    def _all_same_class(self, items):
        """Raise TypeError if items are not all Windows or all Tranforms"""
        error_msg = (
            "Pipeline items must be either: Window, BaseTransform, "
            "List[Window], or List[BaseTransform]"
        )

        if not isinstance(items[0], (Window, BaseTransform)):
            raise TypeError(error_msg)

        if isinstance(items[0], Window):
            for item in items:
                if not isinstance(item, Window):
                    raise TypeError(error_msg)

        if isinstance(items[0], BaseTransform):
            for item in items:
                if not isinstance(item, BaseTransform):
                    raise TypeError(error_msg)
