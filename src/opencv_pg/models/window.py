import copy
import logging

import numpy as np
from qtpy import QtCore

log = logging.getLogger(__name__)


class Window(QtCore.QObject):
    image_updated = QtCore.Signal()

    counter = 1

    def __init__(self, transforms, name: str = ""):
        super().__init__()
        self.transforms = transforms
        self.index = None
        self.pipeline = None
        self.last_out = None

        # Use a suitable name if none is provided
        if not name:
            self.name = f"Step {self.counter}"
            Window.counter += 1
        else:
            self.name = name

    def start_pipeline(self, transform_index: int = 0):
        """Runs pipeline from current window, starting on `transform_index`"""
        log.debug(
            "Starting Pipeline from Window %s, transform index: %s",
            self.index,
            transform_index,
        )
        self.pipeline.run_pipeline(self.index, transform_index)

    def draw(self, img_in, extra_in, transform_index=0):
        """Call _draw on each child transform in sequence and return final output"""
        if transform_index < 0:
            raise ValueError(f"Transform index must be >= 0. Got {transform_index}")

        if img_in is not None and len(img_in.shape) > 0:
            self.last_in = np.copy(img_in)
            img_out = np.copy(img_in)
            self.extra_in = copy.deepcopy(extra_in)
            extra_out = copy.deepcopy(extra_in)
        else:
            img_out = None
            extra_out = None

        # Run the transforms
        for transform in self.transforms[transform_index:]:
            img_out, extra_out = transform._draw(img_out, extra_out)

        self.last_out = np.copy(img_out)
        self.image_updated.emit()
        return img_out, extra_out
