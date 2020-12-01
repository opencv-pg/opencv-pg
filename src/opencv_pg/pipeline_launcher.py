import logging

from qtpy import QtWidgets

from opencv_pg.models.pipeline import Pipeline
from opencv_pg.views.pipeline_window import PipeWindow

LOG_FORMAT = "%(levelname)s:%(name)s:%(lineno)d:%(message)s"

DEF_SPLIT_PERCENTAGE = 0.5
DEF_WINDOW_SIZE = (400, 600)
DEF_OFFSET = 30


def launch_pipeline(pipeline: Pipeline, show_info_widgets: bool = False,
                    log_level=logging.INFO):
    """Opens a PipelineWindow for each Window in the Pipeline

    Args:
        pipeline (Pipeline): Incoming Pipeline
        show_info_widgets (bool): If True, shows info_widgets on Transforms.
            Default is False.
        log_level (logging.LEVEL_NAME): Log level to use. Default is INFO.
    """
    logging.basicConfig(level=log_level, format=LOG_FORMAT)
    app = QtWidgets.QApplication([])

    # Have to keep windows in scope to show them, so store them in a list
    windows = []
    for window in pipeline.windows:
        pipe_win = PipeWindow(window, show_info_widget=show_info_widgets)
        pipe_win.resize(*DEF_WINDOW_SIZE)
        pipe_win.setWindowTitle(window.name)
        windows.append(pipe_win)
        pipe_win.show()

    # Run pipeline after creating windows so they repaint themselves
    pipeline.run_pipeline()

    # Reset each viewer so image is fit to its size
    last_pos = None
    for win in windows:
        win.set_splitter_pos(DEF_SPLIT_PERCENTAGE)
        win.viewer.reset_view()

        # Stagger windows
        if last_pos is None:
            last_pos = (win.x(), win.y())
        else:
            win.move(last_pos[0] + DEF_OFFSET, last_pos[1] + DEF_OFFSET)
            last_pos = (win.x(), win.y())
    app.exec_()
