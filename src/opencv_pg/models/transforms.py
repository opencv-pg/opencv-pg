import logging

from .base_transform import BaseTransform
from . import params
from . import cv2_constants as cvc
from . import support_transforms as supt

from qtpy import QtWidgets, QtCore
import cv2
import numpy as np

ROUND_PI = np.round(np.pi, 3)

log = logging.getLogger(__name__)


class GaussianBlur(BaseTransform):
    k_size_x = params.IntSlider(min_val=1, max_val=100, default=1, step=2)
    k_size_y = params.IntSlider(min_val=1, max_val=100, default=1, step=2)
    sigma_x = params.IntSlider(min_val=1, max_val=10, default=1)
    sigma_y = params.IntSlider(min_val=0, max_val=10, default=0)
    border_type = params.ComboBox(
        options=[
            "BORDER_CONSTANT",
            "BORDER_REPLICATE",
            "BORDER_REFLECT",
            "BORDER_DEFAULT",
            "BORDER_ISOLATED",
        ],
        default="BORDER_DEFAULT",
        options_map=cvc.BORDERS,
    )

    def draw(self, img_in, extra_in):
        return cv2.GaussianBlur(
            src=img_in,
            ksize=(self.k_size_x, self.k_size_y),
            sigmaX=self.sigma_x,
            sigmaY=self.sigma_y,
            borderType=self.border_type,
        )


class MedianBlur(BaseTransform):
    k_size = params.IntSlider(min_val=1, max_val=100, default=11, step=2)

    def draw(self, img_in, extra_in):
        return cv2.medianBlur(img_in, self.k_size)


class CopyMakeBorder(BaseTransform):
    doc_filename = "copyMakeBorder.html"

    top = params.IntSlider(min_val=0, max_val=50, default=30)
    right = params.IntSlider(min_val=0, max_val=50, default=30)
    bottom = params.IntSlider(min_val=0, max_val=50, default=30)
    left = params.IntSlider(min_val=0, max_val=50, default=30)
    border_type = params.ComboBox(
        options=[
            "BORDER_CONSTANT",
            "BORDER_REPLICATE",
            "BORDER_REFLECT",
            "BORDER_WRAP",
            "BORDER_DEFAULT",
            "BORDER_ISOLATED",
        ],
        default="BORDER_DEFAULT",
        options_map=cvc.BORDERS,
    )
    border_val = params.ColorPicker(label="Value")

    def draw(self, img_in, extra_in):
        kwargs = dict(
            src=img_in,
            top=self.top,
            bottom=self.bottom,
            left=self.left,
            right=self.right,
            borderType=self.border_type,
        )
        if self.border_type == cv2.BORDER_CONSTANT:
            kwargs["value"] = self.border_val
        return cv2.copyMakeBorder(**kwargs)

    def update_widgets_state(self):
        """Enable/Disable border_val based on selected borderType"""
        if self.border_type == cv2.BORDER_CONSTANT:
            self._border_val.set_enabled(True)
        else:
            self._border_val.set_enabled(False)


class Normalize(BaseTransform):
    doc_filename = "normalize.html"

    alpha = params.IntSlider(default=125, min_val=0, max_val=255, step=1)
    beta = params.IntSlider(default=0, min_val=0, max_val=255)
    norm_type = params.ComboBox(
        options=["NORM_MINMAX", "NORM_INF", "NORM_L1", "NORM_L2"],
        default="NORM_L2",
        options_map=cvc.NORMS,
    )

    def draw(self, img_in, extra_in):
        # cv2 seems to require dst; throws error when not provided
        ret = cv2.normalize(
            src=img_in,
            dst=None,
            alpha=self.alpha,
            beta=self.beta,
            norm_type=self.norm_type,
            dtype=-1,
        )
        return ret

    def update_widgets_state(self):
        if self.norm_type == cv2.NORM_L1:
            self._beta.set_enabled(False)
            limit = self.last_in.size * 255
            self._alpha.set_max(limit)
            self._alpha.set_step(limit // 500)
        elif self.norm_type == cv2.NORM_L2:
            self._beta.set_enabled(False)
            limit = np.sqrt(self.last_in.size) * 255
            self._alpha.set_max(limit)
            self._alpha.set_step(limit // 500)
        elif self.norm_type in (cv2.NORM_INF, cv2.NORM_MINMAX):
            self._alpha.set_max(255)
            self._alpha.set_step(1)
            if self.norm_type == cv2.NORM_MINMAX:
                self._beta.set_enabled(True)
            else:
                self._beta.set_enabled(False)
        self._alpha._value = self._alpha.widget.slider.value()


class Split(BaseTransform):
    doc_filename = "split.html"

    def draw(self, img_in, extra_in):
        out = cv2.split(img_in)
        return img_in, out

    def get_info_widget(self):
        """Adds labels centered under the images describing the channel"""
        wid = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()
        left = QtWidgets.QLabel("Blue Channel", alignment=QtCore.Qt.AlignLeft)
        mid = QtWidgets.QLabel("Green Channel", alignment=QtCore.Qt.AlignCenter)
        right = QtWidgets.QLabel("Red Channel", alignment=QtCore.Qt.AlignRight)
        layout.addWidget(left)
        layout.addWidget(mid)
        layout.addWidget(right)
        wid.setLayout(layout)
        return wid


class Merge(BaseTransform):
    doc_filename = "merge.html"

    def draw(self, img_in, extra_in):
        """Since need to merge multiple channels to 1, first split, then merge"""
        merged = cv2.merge(extra_in)
        return img_in, merged


class Filter2D(BaseTransform):
    doc_filename = "filter2D.html"

    kernel = params.Array(help_text="Right click in array to set anchor")
    delta = params.SpinBox(min_val=0, max_val=300, step=5, unit_type="integer")
    border_type = params.ComboBox(
        options=[
            "BORDER_CONSTANT",
            "BORDER_REPLICATE",
            "BORDER_REFLECT",
            "BORDER_DEFAULT",
            "BORDER_ISOLATED",
        ],
        default="BORDER_DEFAULT",
        options_map=cvc.BORDERS,
    )

    def draw(self, img_in, extra_in):
        out = cv2.filter2D(
            src=img_in,
            ddepth=-1,
            kernel=self.kernel,
            anchor=self._kernel.anchor,
            delta=self.delta,
            borderType=self.border_type,
        )
        return out


class Canny(BaseTransform):
    threshold1 = params.FloatSlider(min_val=0, max_val=1000, default=0)
    threshold2 = params.FloatSlider(min_val=0, max_val=1000, default=0)
    aperture_size = params.IntSlider(min_val=3, max_val=7, step=2, default=3)
    use_l2_gradient = params.CheckBox()

    def draw(self, img_in, extra_in):
        img = supt.make_gray(img_in)
        out = cv2.Canny(
            image=img,
            threshold1=self.threshold1,
            threshold2=self.threshold2,
            apertureSize=self.aperture_size,
            L2gradient=self.use_l2_gradient,
        )
        return out


class HoughLines(BaseTransform):
    rho = params.FloatSlider(min_val=1, max_val=200, default=1.0)
    theta = params.FloatSlider(min_val=0.01, max_val=ROUND_PI, step=0.01, default=1)
    threshold = params.IntSlider(min_val=0, max_val=250, default=1)
    srn = params.IntSlider(min_val=0, max_val=300, default=0, step=1)
    stn = params.IntSlider(min_val=0, max_val=300, default=0, step=1)
    min_theta = params.FloatSlider(min_val=0, max_val=ROUND_PI, default=0, step=0.05)
    max_theta = params.FloatSlider(
        min_val=0, max_val=ROUND_PI, default=ROUND_PI, step=0.05
    )

    def draw(self, img_in, extra_in):
        # Docs indicate image may be modified by the function
        supt.is_gray_uint8(img_in)
        img = np.copy(img_in)
        lines = cv2.HoughLines(
            image=img,
            rho=self.rho,
            theta=self.theta,
            threshold=self.threshold,
            srn=self.srn,
            stn=self.stn,
            min_theta=self.min_theta,
            max_theta=self.max_theta,
        )
        if lines is not None:
            inner, outer = lines.shape[0], lines.shape[-1]
            lines = lines.reshape((inner, outer))
        return img_in, lines


class HoughLinesP(BaseTransform):
    rho = params.FloatSlider(min_val=1, max_val=500, default=1)
    theta = params.FloatSlider(min_val=0.01, max_val=ROUND_PI, step=0.01, default=1)
    threshold = params.IntSlider(min_val=0, max_val=500, default=1)
    min_length = params.IntSlider(min_val=0, max_val=500, default=100)
    max_gap = params.IntSlider(min_val=0, max_val=500, default=100)

    def draw(self, img_in, extra_in):
        # Docs indicate image may be modified by the function
        supt.is_gray_uint8(img_in)
        img = np.copy(img_in)
        lines = cv2.HoughLinesP(
            image=img,
            rho=self.rho,
            theta=self.theta,
            threshold=self.threshold,
            minLineLength=self.min_length,
            maxLineGap=self.max_gap,
        )
        # Remove nuisance inner array
        if lines is not None:
            inner, outer = lines.shape[0], lines.shape[-1]
            lines = lines.reshape((inner, outer))
        return img_in, lines


class HoughCircles(BaseTransform):
    method = params.ComboBox(
        options=["HOUGH_GRADIENT", "HOUGH_GRADIENT_ALT"],
        options_map=cvc.HOUGH,
        default="HOUGH_GRADIENT",
    )
    dp = params.FloatSlider(min_val=1, max_val=5, default=1, step=0.01)
    min_dist = params.IntSlider(min_val=1, max_val=500, default=50)
    param1 = params.IntSlider(min_val=1, max_val=200, default=1, step=1)
    param2 = params.FloatSlider(min_val=1, max_val=200, default=10, step=1)
    min_radius = params.IntSlider(min_val=1, max_val=250, default=50)
    max_radius = params.IntSlider(min_val=-1, max_val=250, default=100)

    def update_widgets_state(self):
        if self.method == cv2.HOUGH_GRADIENT_ALT:
            self._param1.set_min(0)
            self._param1.set_max(350)
            self._param1.set_step(1)
            self._param2.set_min(0.0)
            self._param2.set_max(1.0)
            self._param2.set_step(1 / 500)
        else:
            self._param2.set_min(1.0)
            self._param2.set_max(200)
            self._param2.set_step(1)
            self._param1.set_min(1)
            self._param1.set_max(200)
            self._param1.set_step(1)

    def draw(self, img_in, extra_in):
        img = supt.make_gray(img_in)
        circles = cv2.HoughCircles(
            image=img,
            method=self.method,
            dp=self.dp,
            minDist=self.min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )
        return img_in, circles[0] if circles is not None else None


class Dilate(BaseTransform):
    doc_filename = "dilate.html"

    kernel = params.Array(
        use_struct=True, help_text="Right click in array to set anchor"
    )
    iterations = params.IntSlider(min_val=1, max_val=100, default=1)
    border_type = params.ComboBox(
        options=[
            "BORDER_CONSTANT",
            "BORDER_REPLICATE",
            "BORDER_REFLECT",
            "BORDER_DEFAULT",
            "BORDER_ISOLATED",
        ],
        default="BORDER_DEFAULT",
        options_map=cvc.BORDERS,
    )
    border_val = params.ColorPicker(label="Value")

    def update_widgets_state(self):
        """Enable/Disable border_val based on selected borderType"""
        if self.border_type == cv2.BORDER_CONSTANT:
            self._border_val.set_enabled(True)
        else:
            self._border_val.set_enabled(False)

    def draw(self, img_in, extra_in):
        kwargs = dict(
            src=img_in,
            kernel=self.kernel.astype(np.uint8),
            anchor=self._kernel.anchor,
            iterations=self.iterations,
            borderType=self.border_type,
        )
        if self.border_type == cv2.BORDER_CONSTANT:
            kwargs["borderValue"] = self.border_val
        out = cv2.dilate(**kwargs)
        return out


class BilateralFilter(BaseTransform):
    doc_filename = "bilateralFilter.html"

    d = params.IntSlider(min_val=0, max_val=12, default=5, label="Diameter")
    sigma_color = params.IntSlider(min_val=0, max_val=200, default=100)
    sigma_space = params.IntSlider(min_val=0, max_val=200, default=100)
    border_type = params.ComboBox(
        options=[
            "BORDER_CONSTANT",
            "BORDER_REPLICATE",
            "BORDER_REFLECT",
            "BORDER_WRAP",
            "BORDER_DEFAULT",
            "BORDER_ISOLATED",
        ],
        default="BORDER_DEFAULT",
        options_map=cvc.BORDERS,
    )

    def draw(self, img_in, extra_in):
        out = cv2.bilateralFilter(
            src=img_in,
            d=self.d,
            sigmaColor=self.sigma_color,
            sigmaSpace=self.sigma_space,
            borderType=self.border_type,
        )
        return out


class Sobel(BaseTransform):
    doc_filename = "Sobel.html"

    dx = params.IntSlider(min_val=1, max_val=2, default=1, step=1)
    dy = params.IntSlider(min_val=1, max_val=2, default=1, step=1)
    k_size = params.IntSlider(min_val=1, max_val=7, default=3, step=2)
    scale = params.FloatSlider(min_val=0.10, max_val=200, default=100.0, step=0.1)
    delta = params.IntSlider(min_val=-255, max_val=255, default=0)
    border_type = params.ComboBox(
        options=[
            "BORDER_CONSTANT",
            "BORDER_REPLICATE",
            "BORDER_REFLECT",
            "BORDER_DEFAULT",
            "BORDER_ISOLATED",
        ],
        default="BORDER_DEFAULT",
        options_map=cvc.BORDERS,
    )

    def draw(self, img_in, extra_in):
        out = cv2.Sobel(
            src=img_in,
            ddepth=-1,
            dx=self.dx,
            dy=self.dy,
            ksize=self.k_size,
            scale=self.scale,
            delta=self.delta,
            borderType=self.border_type,
        )
        return out


class Remap(BaseTransform):
    doc_filename = "remap.html"

    r = params.IntSlider(min_val=1, max_val=50, default=30, step=1)
    theta = params.IntSlider(min_val=1, max_val=90, default=20, step=1)
    interpolation_type = params.ComboBox(
        options=["INTER_NEAREST", "INTER_LINEAR", "INTER_CUBIC", "INTER_LANCZOS4",],
        default="INTER_LINEAR",
        options_map=cvc.INTERPOLATION,
    )
    border_type = params.ComboBox(
        options=[
            "BORDER_CONSTANT",
            "BORDER_REPLICATE",
            "BORDER_REFLECT",
            "BORDER_WRAP",
            "BORDER_DEFAULT",
            "BORDER_TRANSPARENT",
        ],
        default="BORDER_DEFAULT",
        options_map=cvc.BORDERS,
    )
    border_val = params.ColorPicker(label="Value")

    def update_widgets_state(self):
        """Enable/Disable border_val based on selected borderType"""
        if self.border_type == cv2.BORDER_CONSTANT:
            self._border_val.set_enabled(True)
        else:
            self._border_val.set_enabled(False)

    def get_info_widget(self):
        label = QtWidgets.QLabel(
            "Apply the mapping functions: \n"
            "map1(x,y)=x+r*cos(theta*x/num_cols) \n"
            "map2(x,y)=y+r*cos(theta*y/num_rows) ",
            alignment=QtCore.Qt.AlignCenter,
        )
        label.setWordWrap(True)
        return label

    def draw(self, img_in, extra_in):
        num_rows = img_in.shape[1]
        num_cols = img_in.shape[0]
        xmap = np.tile(np.arange(num_cols, dtype=np.float32), (num_rows, 1))
        ymap = np.tile(
            np.arange(num_rows, dtype=np.float32).reshape((-1, 1)), (1, num_cols)
        )
        theta_x = self.r * np.cos(self.theta * xmap / num_cols)
        theta_y = self.r * np.sin(self.theta * ymap / num_rows)
        xmap += theta_x
        ymap += theta_y

        kwargs = dict(
            src=img_in,
            map1=xmap,
            map2=ymap,
            interpolation=self.interpolation_type,
            borderMode=self.border_type,
        )
        if self.border_type == cv2.BORDER_CONSTANT:
            kwargs["borderValue"] = self.border_val
        return cv2.remap(**kwargs)


class SepFilter2D(BaseTransform):
    doc_filename = "sepFilter2D.html"

    kernel_X = params.Array(
        use_struct=False, dims=1, help_text="Right click in array to set anchor"
    )
    kernel_Y = params.Array(
        use_struct=False, dims=1, help_text="Right click in array to set anchor"
    )
    delta = params.IntSlider(min_val=-255, max_val=255, default=0, label="Delta")
    border_type = params.ComboBox(
        options=[
            "BORDER_CONSTANT",
            "BORDER_REPLICATE",
            "BORDER_REFLECT",
            "BORDER_DEFAULT",
            "BORDER_ISOLATED",
        ],
        default="BORDER_DEFAULT",
        options_map=cvc.BORDERS,
    )

    def draw(self, img_in, extra_in):
        out = cv2.sepFilter2D(
            src=img_in,
            ddepth=-1,
            kernelX=self.kernel_X[0],
            kernelY=self.kernel_Y[0],
            anchor=(self._kernel_X.anchor, self._kernel_Y.anchor),
            delta=self.delta,
            borderType=self.border_type,
        )
        return out


class BoxFilter(BaseTransform):
    doc_filename = "boxFilter.html"

    kernel = params.Array(
        help_text="Right click in array to set anchor", editable_array=False
    )
    normalize = params.CheckBox()
    border_type = params.ComboBox(
        options=[
            "BORDER_CONSTANT",
            "BORDER_REPLICATE",
            "BORDER_REFLECT",
            "BORDER_DEFAULT",
            "BORDER_ISOLATED",
        ],
        default="BORDER_DEFAULT",
        options_map=cvc.BORDERS,
    )

    def interconnect_widgets(self):
        self._kernel.widget.array_size.valueChanged.connect(
            self._handle_dimensions_changed
        )
        self._normalize.widget.stateChanged.connect(self._handle_checkbox_changed)

    def common_handler(self, rows, cols):
        if self.normalize:
            ar = np.ones((rows, cols)) / (rows * cols)
        else:
            ar = np.ones((rows, cols))
        self._kernel.widget.array.model().set_internal_model_data(ar)

    @QtCore.Slot(int, int)
    def _handle_dimensions_changed(self, rows, cols):
        self.common_handler(rows, cols)

    @QtCore.Slot(int)
    def _handle_checkbox_changed(self, state):
        rows, cols = self.kernel.shape
        self.common_handler(rows, cols)

    def draw(self, img_in, extra_in):
        out = cv2.boxFilter(
            src=img_in,
            ddepth=-1,
            ksize=self.kernel.shape,
            anchor=self._kernel.anchor,
            normalize=self.normalize,
            borderType=self.border_type,
        )
        return out


class FastNIMeansDenoisingColored(BaseTransform):
    doc_filename = "fastNIMeansDenoisingColored.html"

    template_window_size = params.IntSlider(min_val=1, max_val=15, default=7, step=2)
    search_window_size = params.IntSlider(min_val=1, max_val=51, default=21, step=2)
    h = params.FloatSlider(min_val=0, max_val=25, default=3)
    h_color = params.FloatSlider(min_val=0, max_val=25, default=7)

    def draw(self, img_in, extra_in):
        out = cv2.fastNlMeansDenoisingColored(
            src=img_in,
            templateWindowSize=self.template_window_size,
            searchWindowSize=self.search_window_size,
            h=self.h,
            hColor=self.h_color,
        )
        return out


class Kmeans(BaseTransform):
    doc_name = "kmeans.html"

    k = params.IntSlider(
        min_val=1, max_val=10, default=5, help_text="Number of Clusters"
    )
    criteria = params.ComboBox(
        options=[
            "TERM_CRITERIA_EPS",
            "TERM_CRITERIA_MAX_ITER",
            "EPS + MAX_ITER (Either)",
        ],
        options_map=cvc.TERM,
    )
    epsilon = params.FloatSlider(min_val=0.001, max_val=0.2, step=0.005, default=0.005)
    max_iter = params.IntSlider(min_val=1, max_val=200, default=10)
    attempts = params.IntSlider(min_val=1, max_val=100, default=1)
    flags = params.ComboBox(
        options=["KMEANS_RANDOM_CENTERS", "KMEANS_PP_CENTERS"], options_map=cvc.KFLAGS
    )

    def update_widgets_state(self):
        """Disable/Enable epsilon/max_iter based on term_type"""
        if self.criteria == cv2.TERM_CRITERIA_EPS:
            self._max_iter.widget.setEnabled(False)
            self._epsilon.widget.setEnabled(True)
        elif self.criteria == cv2.TERM_CRITERIA_MAX_ITER:
            self._max_iter.widget.setEnabled(True)
            self._epsilon.widget.setEnabled(False)
        else:
            self._max_iter.widget.setEnabled(True)
            self._epsilon.widget.setEnabled(True)

    def draw(self, img_in, extra_in):
        points = extra_in.astype(np.float32)
        comp, labels, centers = cv2.kmeans(
            points,
            K=self.k,
            bestLabels=None,
            criteria=(self.criteria, self.max_iter, self.epsilon),
            attempts=self.attempts,
            flags=self.flags,
        )
        return img_in, (points, labels, centers)


class InRange(BaseTransform):
    """An InRange with assumed previous CvtColor and post bitwise_and

    This specific implementation assumes the Transform at index 1 is a CvtColor.
    and that a bitwise_and will come next. If you need a raw implemenation of
    InRange, use InRangeRaw.
    """

    doc_filename = "inRange.html"

    ch1 = params.SliderPairParam(min_val=0, max_val=255)
    ch2 = params.SliderPairParam(min_val=0, max_val=255)
    ch3 = params.SliderPairParam(min_val=0, max_val=255)

    def get_info_widget(self):
        label = QtWidgets.QLabel(
            "Since inRange returns what is effectively a bitmask, we can "
            "combine it with a preceeding cvtColor and then trailing bitwise_and "
            "to filter by color, which is done here.",
            alignment=QtCore.Qt.AlignCenter,
        )
        label.setWordWrap(True)
        return label

    def update_widgets_state(self):
        """Update the widget labels to the proper color map"""
        color_range = self.get_transform(1).color_range
        group = self.params[0].widget.parent()
        group.change_label(self._ch1.widget, cvc.COLOR_MAP[color_range][0])
        group.change_label(self._ch2.widget, cvc.COLOR_MAP[color_range][1])
        group.change_label(self._ch3.widget, cvc.COLOR_MAP[color_range][2])

    def draw(self, img_in, extra_in):
        lower = (self.ch1["top"], self.ch2["top"], self.ch3["top"])
        upper = (self.ch1["bot"], self.ch2["bot"], self.ch3["bot"])
        out = cv2.inRange(src=img_in, lowerb=lower, upperb=upper,)
        return extra_in, out


class InRangeRaw(BaseTransform):
    """A raw implemenation of cv2.inRange"""

    doc_filename = "inRange.html"

    ch1 = params.SliderPairParam(min_val=0, max_val=255)
    ch2 = params.SliderPairParam(min_val=0, max_val=255)
    ch3 = params.SliderPairParam(min_val=0, max_val=255)

    def draw(self, img_in, extra_in):
        lower = (self.ch1["top"], self.ch2["top"], self.ch3["top"])
        upper = (self.ch1["bot"], self.ch2["bot"], self.ch3["bot"])
        out = cv2.inRange(src=img_in, lowerb=lower, upperb=upper,)
        return out


class CornerHarris(BaseTransform):
    """Harris Corners"""

    doc_filename = "cornerHarris.html"

    block_size = params.IntSlider(min_val=1, max_val=100, default=1)
    ksize = params.IntSlider(min_val=1, max_val=7, default=3, step=2)
    k = params.FloatSlider(min_val=0.005, max_val=1, default=0.1, step=0.005)
    border_type = params.ComboBox(
        options=[
            "BORDER_CONSTANT",
            "BORDER_REPLICATE",
            "BORDER_REFLECT",
            "BORDER_DEFAULT",
            "BORDER_ISOLATED",
        ],
        default="BORDER_DEFAULT",
        options_map=cvc.BORDERS,
    )

    def draw(self, img_in, extra_in):
        values = cv2.cornerHarris(
            src=img_in,
            blockSize=self.block_size,
            ksize=self.ksize,
            k=self.k,
            borderType=self.border_type,
        )
        return img_in, values


class CornerSubPix(BaseTransform):
    """cornerSubPix"""

    doc_filename = "cornerSubPix.html"

    window_size_rows = params.IntSlider(min_val=1, max_val=100, default=5)
    window_size_cols = params.IntSlider(min_val=1, max_val=100, default=5)
    criteria = params.ComboBox(
        options=[
            "TERM_CRITERIA_EPS",
            "TERM_CRITERIA_MAX_ITER",
            "EPS + MAX_ITER (Either)",
        ],
        options_map=cvc.TERM,
    )
    epsilon = params.FloatSlider(min_val=0.001, max_val=0.2, step=0.005, default=0.005)
    max_iter = params.IntSlider(min_val=1, max_val=200, default=10)

    def update_widgets_state(self):
        """Disable/Enable epsilon/max_iter based on term_type"""
        if self.criteria == cv2.TERM_CRITERIA_EPS:
            self._max_iter.widget.setEnabled(False)
            self._epsilon.widget.setEnabled(True)
        elif self.criteria == cv2.TERM_CRITERIA_MAX_ITER:
            self._max_iter.widget.setEnabled(True)
            self._epsilon.widget.setEnabled(False)
        else:
            self._max_iter.widget.setEnabled(True)
            self._epsilon.widget.setEnabled(True)

    def draw(self, img_in, extra_in):
        # TODO: Add DocString

        img = supt.make_gray(img_in)
        corners = extra_in

        if corners is None:
            return img_in

        # NOTE: cornerSubPix uses an In/Out so it updates whatever you send into
        # corners
        updated_corners = np.copy(corners)
        updated_corners = cv2.cornerSubPix(
            image=img,
            corners=updated_corners,
            winSize=(self.window_size_rows, self.window_size_cols),
            zeroZone=(-1, -1),
            criteria=(self.criteria, self.max_iter, self.epsilon),
        )

        return img_in, (corners, updated_corners)


class GoodFeaturesToTrack(BaseTransform):
    """goodFeaturesToTrack"""

    doc_filename = "goodFeaturesToTrack.html"

    max_corners = params.IntSlider(min_val=0, max_val=100, default=0)
    quality_level = params.FloatSlider(min_val=0.001, max_val=1.0, default=0.1, step=0.001)
    min_distance = params.FloatSlider(min_val=0, max_val=100, default=5, step=0.001)
    block_size = params.IntSlider(min_val=1, max_val=7, default=3, step=2)
    use_harris_detector = params.CheckBox()
    k = params.FloatSlider(min_val=0.005, max_val=1, default=0.1, step=0.005)

    def update_widgets_state(self):
        """Enable/Disable k based on selected HarrisDetector"""
        if self.use_harris_detector:
            self._k.set_enabled(True)
        else:
            self._k.set_enabled(False)

    def draw(self, img_in, extra_in):

        img = supt.make_gray(img_in)
        kwargs = dict(
            image=img,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size,
            useHarrisDetector=self.use_harris_detector,
        )
        if self.use_harris_detector:
            kwargs["k"] = self.k
        corners = cv2.goodFeaturesToTrack(**kwargs)

        return img_in, corners


class Resize(BaseTransform):
    """Resize"""

    doc_filename = "resize.html"

    scale_x = params.FloatSlider(min_val=0.005, max_val=3.0, default=1.0, step=0.005)
    scale_y = params.FloatSlider(min_val=0.005, max_val=3.0, default=1.0, step=0.005)
    interpolation_type = params.ComboBox(
        options=[
            "INTER_NEAREST",
            "INTER_LINEAR",
            "INTER_AREA",
            "INTER_CUBIC",
            "INTER_LANCZOS4",
        ],
        default="INTER_LINEAR",
        options_map=cvc.INTERPOLATION,
    )

    def draw(self, img_in, extra_in):
        out = cv2.resize(
            src=img_in,
            dsize=(0, 0),
            fx=self.scale_x,
            fy=self.scale_y,
            interpolation=self.interpolation_type,
        )
        return out


class ApproxPolyDP(BaseTransform):
    """ApproxPolyDP"""

    doc_filename = "approxPolyDP.html"

    epsilon = params.FloatSlider(min_val=0.005, max_val=30.0, default=1.0, step=0.005)
    closed = params.CheckBox()

    def draw(self, img_in, extra_in):

        img = supt.make_gray(img_in)

        contours = extra_in
        approx_contours = []

        for cont in contours:
            epsilon = self.epsilon
            approx = cv2.approxPolyDP(cont, epsilon, self.closed)
            approx_contours.append(approx)

        return img_in, approx_contours


class FindContours(BaseTransform):
    """FindContours"""

    doc_filename = "findContours.html"

    threshold = params.IntSlider(min_val=0, max_val=255, default=100, step=1)
    mode = params.ComboBox(
        options=["RETR_EXTERNAL", "RETR_LIST", "RETR_CCOMP", "RETR_TREE",],
        default="RETR_CCOMP",
        options_map=cvc.RETR,
    )
    method = params.ComboBox(
        options=[
            "CHAIN_APPROX_NONE",
            "CHAIN_APPROX_SIMPLE",
            "CHAIN_APPROX_TC89_L1",
            "CHAIN_APPROX_TC89_KCOS",
        ],
        default="CHAIN_APPROX_SIMPLE",
        options_map=cvc.CHAIN_APPROX,
    )

    def get_info_widget(self):
        label = QtWidgets.QLabel(
            "We first apply a threshold to the image before searching for contours. ",
            alignment=QtCore.Qt.AlignCenter,
        )
        label.setWordWrap(True)
        return label

    def draw(self, img_in, extra_in):

        img = supt.make_gray(img_in)

        white = img >= self.threshold
        img[white] = 255
        img[~white] = 0

        contours, _ = cv2.findContours(image=img, mode=self.mode, method=self.method)

        return img_in, contours


class GetGaussianKernel(BaseTransform):
    """GetGaussianKernel"""

    doc_filename = "getGaussianKernel.html"

    k_size = params.IntSlider(min_val=1, max_val=31.0, default=13, step=2)
    sigma = params.FloatSlider(min_val=-1.0, max_val=31.0, default=13.0, step=0.005)

    def get_info_widget(self):
        label = QtWidgets.QLabel(
            "This display is showing the 1D Gaussian kernel as a vertical image. ",
            alignment=QtCore.Qt.AlignCenter,
        )
        label.setWordWrap(True)
        return label

    def draw(self, img_in, extra_in):
        out = cv2.getGaussianKernel(ksize=self.k_size, sigma=self.sigma)
        return img_in, out


class MatchTemplate(BaseTransform):
    """MatchTemplate"""

    doc_filename = "matchTemplate.html"

    template_center_x = params.IntSlider(min_val=0, max_val=100, default=50, step=1)
    template_center_y = params.IntSlider(min_val=0, max_val=100, default=50, step=1)
    template_size = params.IntSlider(min_val=0, max_val=100, default=20, step=1)

    method = params.ComboBox(
        options=[
            "TM_SQDIFF",
            "TM_SQDIFF_NORMED",
            "TM_CCORR",
            "TM_CCORR_NORMED",
            "TM_CCOEFF",
            "TM_CCOEFF_NORMED",
        ],
        default="TM_SQDIFF",
        options_map=cvc.TM,
    )

    def get_info_widget(self):
        label = QtWidgets.QLabel(
            "Selected template shown in red. \nBest template match shown in blue.",
            alignment=QtCore.Qt.AlignCenter,
        )
        label.setWordWrap(True)
        return label

    def draw(self, img_in, extra_in):

        rows, cols = img_in.shape[0:2]
        temp_center = (
            int(rows * self.template_center_y / 100),
            int(cols * self.template_center_x / 100),
        )
        temp_row_size = int(rows * self.template_size / 200)
        temp_col_size = int(cols * self.template_size / 200)

        template = img_in[
            temp_center[0] - temp_row_size : temp_center[0] + temp_row_size,
            temp_center[1] - temp_col_size : temp_center[1] + temp_col_size,
            :,
        ]

        result = cv2.matchTemplate(image=img_in, templ=template, method=self.method)

        cv2.rectangle(
            img_in,
            (temp_center[1] - temp_col_size, temp_center[0] - temp_row_size),
            (temp_center[1] + temp_col_size, temp_center[0] + temp_row_size),
            (0, 0, 255),
            5,
        )

        h, w = template.shape[0:2]
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if self.method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img_in, top_left, bottom_right, 255, 2)

        return img_in


class AddWeighted(BaseTransform):
    """AddWeighted"""

    doc_filename = "addWeighted.html"

    alpha = params.FloatSlider(min_val=0.0, max_val=1.0, default=1.0, step=0.005)
    beta = params.FloatSlider(min_val=0.0, max_val=1.0, default=0.0, step=0.005)
    gamma = params.FloatSlider(min_val=0, max_val=255, default=0.0, step=1.0)

    def draw(self, img_in, extra_in):
        rev_rows = np.arange(img_in.shape[0]) * (-1)
        if len(img_in.shape) == 3:
            img2 = img_in[rev_rows, :, :]
        else:
            img2 = img_in[rev_rows, :]
        out = cv2.addWeighted(
            src1=img_in, alpha=self.alpha, src2=img2, beta=self.beta, gamma=self.gamma
        )
        return out


class CornerEigenValsAndVecs(BaseTransform):
    """CornerEigenValsAndVecs"""

    doc_filename = "cornerEigenValsAndVecs.html"

    block_size = params.IntSlider(min_val=1, max_val=25, default=3)
    k_size = params.IntSlider(min_val=1, max_val=7, default=3, step=2)
    threshold = params.IntSlider(min_val=1, max_val=100, default=50)
    border_type = params.ComboBox(
        options=[
            "BORDER_CONSTANT",
            "BORDER_REPLICATE",
            "BORDER_REFLECT",
            "BORDER_DEFAULT",
            "BORDER_ISOLATED",
        ],
        default="BORDER_DEFAULT",
        options_map=cvc.BORDERS,
    )

    def get_info_widget(self):
        label = QtWidgets.QLabel(
            "This display is using the eigenvalues returned and a threshold "
            "value to build a corner detector. See the OpenCV tutorial for details. ",
            alignment=QtCore.Qt.AlignCenter,
        )
        label.setWordWrap(True)
        return label

    def draw(self, img_in, extra_in):
        img = supt.make_gray(img_in)

        values = cv2.cornerEigenValsAndVecs(
            src=img,
            blockSize=self.block_size,
            ksize=self.k_size,
            borderType=self.border_type,
        )

        Mc = np.empty(img.shape, dtype=np.float32)
        L1 = values[:, :, 0]
        L2 = values[:, :, 1]
        Mc = np.multiply(L1, L2) - 0.04 * (np.multiply((L1 + L2), (L1 + L2)))
        values_min, values_max, _, _ = cv2.minMaxLoc(Mc)
        bound = values_min + (values_max - values_min) * self.threshold / 100

        mask = Mc > bound

        shape_y, shape_x = np.shape(Mc)
        [X, Y] = np.meshgrid(np.arange(shape_x), np.arange(shape_y))
        pts_harris = np.stack((X[mask], Y[mask]), axis=1)

        return img_in, pts_harris


class PyrDown(BaseTransform):
    doc_filename = "pyrDown.html"

    n_images = params.IntSlider(
        min_val=1,
        max_val=5,
        default=3,
        help_text="Number of successive times to run pyrDown",
    )
    border_type = params.ComboBox(
        options=[
            "BORDER_REPLICATE",
            "BORDER_REFLECT",
            "BORDER_WRAP",
            "BORDER_DEFAULT",
        ],
        default="BORDER_DEFAULT",
        options_map=cvc.BORDERS,
    )

    def get_info_widget(self):
        text = (
            "The first image is the original. Each successive image is a "
            "reduction by 1/2 of the previous images dimensions"
        )
        label = QtWidgets.QLabel(text, alignment=QtCore.Qt.AlignCenter)
        label.setWordWrap(True)
        return label

    def draw(self, img_in, extra_in):
        pyramid = []
        pyramid.append(img_in)
        img = img_in
        for _ in range(self.n_images):
            img = cv2.pyrDown(
                src=img,
                dstsize=tuple((int(x / 2) for x in img.shape[:2])),
                borderType=self.border_type,
            )
            pyramid.append(np.copy(img))

        return img_in, pyramid


class FillPoly(BaseTransform):
    doc_filename = "fillPoly.html"

    points = params.ReadOnlyLabel(
        fmt_str="{x}",
        default=np.array(
            [
                # Diamond
                [[125, 0], [250, 125], [125, 250], [0, 125]],
                # Square Inside
                [[75, 75], [175, 75], [175, 175], [75, 175]],
                # Diagonal Rectangle
                [[0, 0], [225, 250], [250, 225], [250, 225],],
            ]
        ),
        help_text="Shapes shown in order are a Diamond, an inner square, and a "
        "diagonal triangle (though four points specify it).",
    )
    color = params.ColorPicker(label="Color", default=(0, 0, 255))
    line_type = params.ComboBox(
        options=list(cvc.LINES.keys()), options_map=cvc.LINES, default="8-Connected"
    )

    def draw(self, img_in, extra_in):
        out = cv2.fillPoly(
            img=img_in, pts=self.points, color=self.color, lineType=self.line_type
        )
        return out

    def get_info_widget(self):
        label = QtWidgets.QLabel(
            "The points shown below are those that define the shape seen",
            alignment=QtCore.Qt.AlignCenter,
        )
        label.setWordWrap(True)
        return label


class Transform(BaseTransform):
    """A do nothing transform with options"""

    slider_1 = params.IntSlider(
        min_val=0, max_val=10, label="Renamed Slider", default=1
    )
    slider_2 = params.IntSlider(
        min_val=10, max_val=20, default=15, editable_range=False
    )
    float_slider = params.FloatSlider(min_val=0, max_val=1, default=0, step=0.1)
    pair = params.SliderPairParam(min_val=0, max_val=255)
    combo = params.ComboBox(
        options=["Option1", "Option2"], options_map={"Option1": 1, "Option2": 2}
    )

    def draw(self, img_in, extra_in):
        return img_in
