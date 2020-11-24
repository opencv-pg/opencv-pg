import logging

from .pipeline import Window
from . import support_transforms as supt
from . import transforms as tf

from opencv_pg.docs.doc_writer import render_local_doc, RENDERED_DIR

log = logging.getLogger(__name__)


_LOADER_CLASS = supt.LoadImage


# Transforms that will populate the BuiltIns transform List
_TRANS_WINDOWS = {
    tf.GaussianBlur: [_LOADER_CLASS, tf.GaussianBlur],
    tf.MedianBlur: [_LOADER_CLASS, tf.MedianBlur],
    tf.CopyMakeBorder: [_LOADER_CLASS, tf.CopyMakeBorder],
    tf.Normalize: [_LOADER_CLASS, tf.Normalize],
    tf.Split: [_LOADER_CLASS, tf.Split],
    tf.Merge: [_LOADER_CLASS, tf.Merge],
    tf.Filter2D: [_LOADER_CLASS, tf.Filter2D],
    tf.Canny: [_LOADER_CLASS, tf.Canny],
    tf.HoughLines: [
        _LOADER_CLASS,
        tf.GaussianBlur,
        tf.Canny,
        tf.HoughLines,
        supt.DrawLinesByPointAndAngle,
    ],
    tf.HoughLinesP: [
        _LOADER_CLASS,
        tf.GaussianBlur,
        tf.Canny,
        tf.HoughLinesP,
        supt.DrawLinesByEndpoints,
    ],
    tf.HoughCircles: [
        _LOADER_CLASS,
        tf.GaussianBlur,
        tf.HoughCircles,
        supt.DrawCircles,
    ],
    tf.Dilate: [_LOADER_CLASS, tf.Dilate],
    tf.BilateralFilter: [_LOADER_CLASS, tf.BilateralFilter],
    tf.SepFilter2D: [_LOADER_CLASS, tf.SepFilter2D],
    tf.FastNIMeansDenoisingColored: [_LOADER_CLASS, tf.FastNIMeansDenoisingColored],
    tf.Kmeans: [supt.ClusterGenerator, tf.Kmeans, supt.DrawKMeansPoints],
    tf.InRange: [_LOADER_CLASS, supt.CvtColor, tf.InRange, supt.BitwiseAnd],
    tf.InRangeRaw: [_LOADER_CLASS, tf.InRangeRaw],
    tf.CornerHarris: [
        _LOADER_CLASS,
        tf.GaussianBlur,
        tf.Canny,
        tf.CornerHarris,
        supt.DisplayHarris,
    ],
    tf.PyrDown: [_LOADER_CLASS, tf.PyrDown],
    tf.FillPoly: [supt.BlankCanvas, tf.FillPoly],
    tf.BoxFilter: [_LOADER_CLASS, tf.BoxFilter],
    tf.Sobel: [_LOADER_CLASS, tf.Sobel],
    tf.Resize: [_LOADER_CLASS, tf.Resize],
    tf.GetGaussianKernel: [_LOADER_CLASS, tf.GetGaussianKernel, supt.DrawGaussianKernel],
    tf.AddWeighted: [_LOADER_CLASS, tf.AddWeighted],
    tf.CornerEigenValsAndVecs: [
        _LOADER_CLASS,
        tf.CornerEigenValsAndVecs,
        supt.DrawCirclesFromPoints,
    ],
    tf.Remap: [_LOADER_CLASS, tf.Remap],
    tf.CornerSubPix: [
        _LOADER_CLASS,
        tf.GoodFeaturesToTrack,
        tf.CornerSubPix,
        supt.DrawCornerSubPix,
    ],
    tf.GoodFeaturesToTrack: [
        _LOADER_CLASS,
        tf.GoodFeaturesToTrack,
        supt.DrawCirclesFromPoints,
    ],
    tf.ApproxPolyDP: [
        _LOADER_CLASS,
        tf.FindContours,
        supt.DrawContours,
        tf.ApproxPolyDP,
        supt.DrawContours,
    ],
    tf.FindContours: [_LOADER_CLASS, tf.FindContours, supt.DrawContours],
    tf.MatchTemplate: [_LOADER_CLASS, tf.MatchTemplate],
}


def collect_builtin_transforms():
    """Return list of Transform subclasses for the builtins tab"""
    transforms = _TRANS_WINDOWS.keys()
    for trans in transforms:
        # Render this now so we can load it later
        render_local_doc(RENDERED_DIR, trans.get_doc_filename())

    transforms = sorted(transforms, key=lambda x: x.__name__)
    return transforms


def init_load(path):
    """Initializer the image loader class so it has the right image

    Args:
        path (Path): Path to file to be loaded
    """
    global _LOADER_CLASS
    # TODO: If it's an image, use LoadImage
    # TODO: If it's a video, use LoadVideo (needs to be created)
    pass


def get_transform_window(transform, img_path):
    """Returns a Pipeline Window for transform `trans_name` using `img_path`

    Args:
        trans_name (str): Name of the Transform Class from models.transforms
        img_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    transforms = _TRANS_WINDOWS.get(transform)
    if transforms is None:
        log.error("Can't find transform %s", transforms)

    loader = None
    if transforms[0] is _LOADER_CLASS:
        loader = _LOADER_CLASS(img_path)

    if loader is None:
        trans_inst = [x() for x in transforms]
    else:
        trans_inst = [loader] + [x() for x in transforms[1:]]
    return Window(trans_inst)
