from unittest import mock

import numpy as np
import pytest

from opencv_pg import Pipeline, Window

from . import transform_helpers as th


@pytest.fixture
def pipeline():
    """Create and return a simple Pipeline

    Each transform will set a ``row``, ``col`` of an ndarray to ``value``

    When run, the successive outputs will be:
    [ # Loader
        [0, 0],
        [0, 0]
    ],
    [ # ValueSetter(0, 0, 1)
        [1, 0],
        [0, 0]
    ],
    [ # ValueSetter(0, 1, 2)
        [1, 2],
        [0, 0]
    ],
    [ # ValueSetter(1, 0, 3)
        [1, 2],
        [3, 0]
    ],
    [ # ValueSetter(1, 1, 4)
        [1, 2],
        [3, 4]
    ],
    """
    pipe = Pipeline([
        Window([
            th.Loader(np.zeros((2, 2))),
            th.ValueSetter(0, 0, 1),
            th.ValueSetter(0, 1, 2),
        ]),
        Window([
            th.ValueSetter(1, 0, 3),
            th.ValueSetter(1, 1, 4),
        ]),
    ])

    return pipe


@pytest.fixture
def window():
    win = Window([
        th.ValueSetter(0, 0, 1),
        th.ValueSetter(0, 1, 2),
        th.ValueSetter(1, 0, 3),
    ])
    return win


@pytest.fixture
def mock_pipe():
    """Returns a pipeline where each Transform is a Mock"""
    pipe = Pipeline([
        Window([
            mock.Mock(name='Loader'),
            mock.Mock(name='ValueSetter'),
            mock.Mock(name='ValueSetter'),
        ]),
        Window([
            mock.Mock(name='ValueSetter'),
            mock.Mock(name='ValueSetter'),
        ]),
    ])

    return pipe


def get_mock_transform():
    m = mock.Mock()
    m._draw.return_value = None, None
    return m


@pytest.fixture
def mock_win():
    """Return Window with two Mock Transforms"""
    with mock.patch('tests.test_models.conftest.th.ValueSetter') as MockVS:
        MockVS.side_effect = get_mock_transform
        return Window([
            MockVS(),
            MockVS(),
            MockVS(),
        ])
