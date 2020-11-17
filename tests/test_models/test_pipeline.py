import numpy as np
from numpy import testing

import pytest


def _change_param(transform, value):
    """Simulate changing a parameter and starting the pipeline"""
    transform.value = value
    transform.start_pipeline()


def test_run_from_start_runs_each_transform(pipeline):
    """Test running pipeline from start gives expected final output"""
    # Given a pipeline
    # When
    out, extra = pipeline.run_pipeline(0, 0)

    # Then
    expected = np.array([[1, 2], [3, 4]])
    testing.assert_array_equal(out, expected)


def test_last_in_correct_for_each_transform(pipeline):
    """last_in should be output from previous transform"""
    # Given a pipeline
    # When
    out, extra = pipeline.run_pipeline(0, 0)

    # Then - last_in should be input from the previous transform
    assert pipeline.get_transform(0, 0).last_in is None

    t0_out = np.array([[0, 0], [0, 0]])
    testing.assert_array_equal(pipeline.get_transform(0, 1).last_in, t0_out)

    t1_out = np.array([[1, 0], [0, 0]])
    testing.assert_array_equal(pipeline.get_transform(0, 2).last_in, t1_out)

    t2_out = np.array([[1, 2], [0, 0]])
    testing.assert_array_equal(pipeline.get_transform(1, 0).last_in, t2_out)

    t3_out = np.array([[1, 2], [3, 0]])
    testing.assert_array_equal(pipeline.get_transform(1, 1).last_in, t3_out)


def test_last_out_of_last_window_matches_pipeline_output(pipeline):
    """pipeline return value should match last_out of final window"""
    # Given a pipeline
    # When
    out, extra = pipeline.run_pipeline(0, 0)

    # Then
    testing.assert_array_equal(
        np.array([[1., 2], [3, 4]]),
        pipeline.windows[-1].last_out
    )


@pytest.mark.parametrize('win_idx, trans_idx, new_val, win0_out, win1_out', (
    (0, 1, 9, np.array([[9., 2], [0, 0]]), np.array([[9., 2], [3, 4]])),
    (0, 2, 9, np.array([[1., 9], [0, 0]]), np.array([[1., 9], [3, 4]])),
    (1, 0, 9, np.array([[1., 2], [0, 0]]), np.array([[1., 2], [9, 4]])),
    (1, 1, 9, np.array([[1., 2], [0, 0]]), np.array([[1., 2], [3, 9]])),
))
def test_changing_params_and_running_pipeline(
        pipeline, win_idx, trans_idx, new_val, win0_out, win1_out):
    """Test changing params at each transform and validating output"""
    # Given - a pipeline has been run once
    out, extra = pipeline.run_pipeline(0, 0)

    # When - a param changes ono a transform
    transform = pipeline.get_transform(win_idx, trans_idx)
    _change_param(transform, new_val)

    # Then - The outputs of each window are properly set
    testing.assert_array_equal(
        pipeline.windows[0].last_out,
        win0_out
    )
    testing.assert_array_equal(
        pipeline.windows[1].last_out,
        win1_out
    )