import numpy as np

from opencv_pg import BaseTransform


class ValueSetter(BaseTransform):

    def __init__(self, row, col, value, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.row = row
        self.col = col
        self.value = value

    def draw(self, img_in, extra_in):
        """Sets row, col to value and returns it"""
        img_in[self.row][self.col] = self.value
        if extra_in is not None:
            extra_in[self.row][self.col] = self.value
        return img_in, extra_in


class Loader(BaseTransform):
    def __init__(self, value, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value = value

    def draw(self, img_in, extra_in):
        return np.copy(self.value)


class DictExtra(BaseTransform):
    def draw(self, img_in, extra_in):
        """Assumes extra_in is dict[dict], adds entry to level 2"""
        extra_in['depth1']['new_entry'] = 1
        return img_in, extra_in