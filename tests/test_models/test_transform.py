import logging
from unittest import mock

import numpy as np
from numpy.testing import assert_array_equal

import pytest

from .transform_helpers import ValueSetter, DictExtra


@pytest.mark.parametrize("img_in", (None, np.copy(None)))
def test_last_in_used_if_img_in_is_none(img_in):
    """Test last_in is used if img_in is None"""
    # Given a Transform with last_in set
    t = ValueSetter(0, 0, 5)
    t.last_in = np.ones((2, 2))
    t.extra_in = np.ones((2, 2)) * 2

    # When draw is called with img_in = None
    # extra in will also be set to t.extra_in when img_in is None
    img_out, extra_out = t._draw(img_in, 5)

    # then the operation will be performed on value in last_in
    expected = np.array([[5, 1], [1, 1]])
    assert_array_equal(img_out, expected)

    # extra_out will be a copy of t.extra_in when img_is None
    expected = np.array([[5, 2], [2, 2]])
    assert_array_equal(extra_out, expected)
    assert extra_out is not t.extra_in


def test_img_in_used_when_not_none_and_saved():
    """When img_in is not none, it's used and saved as last_in"""
    # Given a Transform with last_in set
    t = ValueSetter(0, 0, 5)
    t.last_in = np.ones((2, 2))
    t.extra_in = np.ones((2, 2)) * 2

    # When draw is called with img_in not None and has a shape
    img_in = np.ones((2, 2)) * 3
    extra_in = np.ones((2, 2)) * 4
    img_out, extra_out = t._draw(img_in, extra_in)

    # Then operation is performed on img_in and last_in is set to img_in
    expected = np.array([[5, 3], [3, 3]])
    assert_array_equal(img_out, expected)
    assert_array_equal(t.last_in, img_in)
    assert img_in is not t.last_in

    # extra_in will be stored on t.extra_in as a copy
    expected = np.array([[5, 4], [4, 4]])
    assert_array_equal(extra_out, expected)
    assert_array_equal(t.extra_in, extra_in)


def test_transform_disabled():
    """If transform is disabled, output is a copy of current last_in"""
    # Given a Transform with last_in set that is disabled
    t = ValueSetter(0, 0, 5)
    t.last_in = np.ones((2, 2))
    t.extra_in = np.ones((2, 2)) * 2
    t.enabled = False

    # When draw is called with img_in not None and has a shape
    img_in = np.ones((2, 2)) * 3
    extra_in = np.ones((2, 2)) * 4
    img_out, extra_out = t._draw(img_in, extra_in)

    # Then output will be a copy of last_in
    assert_array_equal(img_out, t.last_in)
    assert_array_equal(extra_out, t.extra_in)

    # id's not equal since it's a copy
    assert img_out is not t.last_in
    assert extra_out is not t.extra_in


def test_extra_in_is_deep_copied():
    """Test that passed to extra_in is deep copied"""
    # given a dict with a dict
    img_in = np.ones((2, 2))
    extra_in = {'depth1': {'two': 2}}

    # And a transfomr that adds a new entry to depth1
    transform = DictExtra()

    # When _draw is called
    img_out, extra_out = transform._draw(img_in, extra_in)

    # Then none of the dict id's are the same as the original
    assert extra_in is not extra_out
    assert extra_in['depth1'] is not extra_out['depth1']
    assert extra_out['depth1']['new_entry'] == 1

    # And the extra_in stored on the transform is a deep copy as well
    assert extra_in is not transform.extra_in
    assert extra_in['depth1'] is not transform.extra_in['depth1']


def test_exception_sets_error_passes_through_incoming_imgage_extra(caplog):
    """On exception in draw, error is not None and img_in/extra_in passed through"""
    caplog.set_level(logging.ERROR)

    # Given image_in/extra_in and a transform
    img_in = np.ones((2, 2))
    extra_in = np.ones((2, 2)) * 2
    transform = ValueSetter(0, 0, 5)
    transform.draw = mock.Mock()
    transform.draw.side_effect = Exception('Bad things, Mikey.')

    # When _draw is called and exception is raised
    img_out, extra_out = transform._draw(img_in, extra_in)

    # Then img_out/extra_out are copies of whatever were passed in
    assert_array_equal(img_in, img_out)
    assert_array_equal(extra_in, extra_out)
    assert transform.error is not None

    # And there is a log output of the exception
    assert len(caplog.records) == 1
    assert caplog.record_tuples == [
        ('opencv_pg.models.base_transform', logging.ERROR, 'Bad things, Mikey.')
    ]

