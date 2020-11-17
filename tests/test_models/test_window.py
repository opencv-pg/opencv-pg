from unittest import mock

import numpy as np
from numpy.testing import assert_array_equal

import pytest


class TestDrawCalledStartDifferentTransforms():
    """Test calling draw when starting on different Transform indexes

    Only the index called and forward should have draw called
    """
    def test_start_at_first_index(self, mock_win):
        """Starting at first index calls draw on all Transforms"""
        # Given a Window, when draw is called with index 0
        mock_win.draw(None, None, 0)

        # Then each transforms has _draw called once
        for transform in mock_win.transforms:
            transform._draw.assert_called_once()

    def test_starting_at_second_index(self, mock_win):
        """Starting at second index calls draw second and last transforms"""
        # Given a Window, when draw is called with index 1
        mock_win.draw(None, None, 1)

        # Then t0 draw isn't called and t1 and t2 are
        t0, t1, t2 = mock_win.transforms
        t0._draw.assert_not_called()
        t1._draw.assert_called_once()
        t2._draw.assert_called_once()

    def test_starting_at_last_index(self, mock_win):
        """Starting at last index calls draw last transform only"""
        # Given a Window, when draw is called with index 2
        mock_win.draw(None, None, 2)

        # Then t0 and t1 draw aren't called and t2 is
        t0, t1, t2 = mock_win.transforms
        t0._draw.assert_not_called()
        t1._draw.assert_not_called()
        t2._draw.assert_called_once()


class TestOutputStartDifferentTransforms():
    @pytest.mark.parametrize(
        ('exp_img, exp_extra, idx'),
        [
            (np.array([[1., 2], [3, 5]]), np.array([[1., 2], [3, 6]]), 0),
            (np.array([[5., 2], [3, 5]]), np.array([[6., 2], [3, 6]]), 1),
            (np.array([[5., 5], [3, 5]]), np.array([[6., 6], [3, 6]]), 2),
        ]
    )
    def test_outputs_after_running_transform_from_idx_once(self, window, exp_img, exp_extra, idx):
        """Should see change from index forward applied to img_in/extra_in"""
        # Given
        img_in = np.ones((2, 2)) * 5
        extra_in = np.ones((2, 2)) * 6

        # When
        img_out, extra_out = window.draw(img_in, extra_in, transform_index=idx)

        # Then
        assert_array_equal(img_out, exp_img)
        assert_array_equal(extra_out, exp_extra)

    @pytest.mark.parametrize(
        ('exp_img, exp_extra, idx, new_val'),
        [
            (np.array([[7., 2], [3, 5]]), np.array([[7., 2], [3, 6]]), 0, 7),
            (np.array([[1., 8], [3, 5]]), np.array([[1., 8], [3, 6]]), 1, 8),
            (np.array([[1., 2], [9, 5]]), np.array([[1., 2], [9, 6]]), 2, 9),
        ]
    )
    def test_outputs_when_changing_values(self, window, exp_img, exp_extra, idx, new_val):
        """Simulate param change by changing transform value and starting from there"""
        # Given - the window has been run once so last input stored in each
        # transform
        img_in = np.ones((2, 2)) * 5
        extra_in = np.ones((2, 2)) * 6
        img_out, extra_out = window.draw(img_in, extra_in, transform_index=0)

        # When - transform started with img_in/extra_in as None, simulating
        # a changed parameter
        window.transforms[idx].value = new_val
        img_out, extra_out = window.draw(None, None, transform_index=idx)

        # Then
        assert_array_equal(img_out, exp_img)
        assert_array_equal(extra_out, exp_extra)
