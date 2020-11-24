import logging
from pathlib import Path

from qtpy import QtWidgets

import argparse
from .main import MainWindow
from .doc_viewer import DocWindow
from .pipeline_launcher import LOG_FORMAT

ROBOT = "robot.jpg"


def get_file_path(rel_path):
    """Return abs path to file [rel_path], relative to __file__ as the root"""
    here = Path(__file__)
    p = Path(rel_path)
    return here.parent.joinpath(p)


def run_playground(args):
    """Run the playground"""
    img_path = args.image
    if img_path is None:
        img_path = get_file_path(ROBOT)
    app = QtWidgets.QApplication([])
    m = MainWindow(img_path, args.no_docs, args.disable_info_widgets)
    m.show()
    app.exec_()


def docview():
    """Launch a WebEngine to view docs"""
    parser = argparse.ArgumentParser("OpenCV DocView")
    parser.add_argument(
        "--template",
        type=str,
        required=False,
        help="Template name in opencv_pg/docs/source_docs folder, eg, GaussianBlur.html",
    )

    args = parser.parse_args()

    app = QtWidgets.QApplication([])
    m = DocWindow(args.template)
    m.show()
    app.exec_()


def _validate_image_path(img_path):
    """Raise FileNotFoundError if img_path doesn't exist"""
    if img_path is not None and not Path(img_path).exists():
        raise FileNotFoundError(img_path)


def main():
    """Application entrypoint"""
    parser = argparse.ArgumentParser("OpenCV Playground")
    parser.add_argument(
        "--image", type=str, help="Path to image to load into playground", default=None
    )
    parser.add_argument(
        "--no-docs",
        action="store_true",
        help="Do not load the doc window",
        default=False,
    )
    parser.add_argument(
        "--disable-info-widgets",
        action="store_true",
        help="Disable all info widgets",
        default=False,
    )
    parser.add_argument(
        "--log-level",
        action="store",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Log Level",
    )
    args = parser.parse_args()

    log_level = logging.getLevelName(args.log_level)
    logging.basicConfig(format=LOG_FORMAT, level=log_level)
    _validate_image_path(args.image)
    run_playground(args)
