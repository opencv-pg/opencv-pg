import numpy as np
import pytest

from opencv_pg.models.base_transform import _break_result_into_parts


def test_input_not_list_or_tuple():
    """When input is not list or tuple, returns input, None"""
    # Given an input
    img_in = np.ones((2, 2))

    # When the function is called with just the input
    img_out, extra_out = _break_result_into_parts(img_in)

    # The output is the input, None
    assert img_out is img_in
    assert extra_out is None


@pytest.mark.parametrize("img_in", (
    (np.ones((2, 2)), ),
    [np.ones((2, 2)), ]
))
def test_input_list_or_tuple_with_len_1(img_in):
    """When input is list or tuple with len 1, returns input, None"""
    # Given an input of tuple or list with len 1
    # When the function is called with just the input
    img_out, extra_out = _break_result_into_parts(img_in)

    # The output is the input, None
    assert img_out is img_in[0]
    assert extra_out is None


@pytest.mark.parametrize("img_in", (
    (np.ones((2, 2)), 'extra_in'),
    [np.ones((2, 2)), 'extra_in']
))
def test_input_is_list_or_tuple_with_len_2(img_in):
    """When input is list or tuple with len 2, returns input, extra_in"""
    # Given an input of tuple or list with len 2
    # When the function is called with (img_in, extra_in)
    img_out, extra_out = _break_result_into_parts(img_in)

    # The output is the input, None
    assert img_out is img_in[0]
    assert extra_out is img_in[1]