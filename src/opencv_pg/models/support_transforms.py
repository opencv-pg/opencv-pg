import logging
from pathlib import Path

import cv2
import numpy as np

from . import params
from . import cv2_constants as cvc
from .base_transform import BaseTransform

log = logging.getLogger(__name__)


def is_gray(img):
    """Return True if this appears to be a single channel (gray) image"""
    if len(img.shape) == 2:
        return True
    if img.shape[2] == 1:
        return True
    return False


def make_gray(img_in):
    """Returns Gray image as in BGR2GRAY if 3 channels detected"""
    shape = img_in.shape
    if len(shape) == 3:
        return cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    return img_in


def is_gray_uint8(img_in):
    """Raises TypeError if img is not gray uint8"""
    if is_gray(img_in) and img_in.dtype == np.uint8:
        return
    raise TypeError(
        "Incoming image must be np.uint8 and have 1 channel. "
        "Got dtype={}, shape={}".format(img_in.dtype, img_in.shape)
    )


class LoadImage(BaseTransform):
    def __init__(self, path: str):
        """Loads and returns an img at path

        Args:
            path (str): path to image file
        """
        super().__init__()
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Path {p} does not exist")
        self.img = cv2.imread(path)

    def draw(self, img, extra):
        return self.img


class DrawLinesByPointAndAngle(BaseTransform):
    """Draws infinte lines from direction and magnitude"""

    color = params.ColorPicker(default=(0, 0, 255))
    thickness = params.IntSlider(min_val=1, max_val=100, default=2)
    line_type = params.ComboBox(
        options=list(cvc.LINES.keys()), options_map=cvc.LINES, default="8-Connected"
    )
    show_points = params.CheckBox(default=True)
    mult = params.SpinBox(
        min_val=0,
        max_val=2000,
        default=1000,
        step=100,
        help_text="Multiplier for line length. Increase if lines "
        "do not extend full length of image.",
        unit_type="integer",
    )

    def draw(self, img_in, extra_in):
        """Draws lines on img_in by a point and angle

        Args:
            img_in (np.ndarray): Image in to draw onto
            extra_in ([(rho, theta), ..]): List of rho, theta pairs, as returned
                by HoughLines

        Returns:
            np.ndarray: Updated image
        """
        img = np.copy(img_in)
        if extra_in is None:
            return img

        b, g, r = self.color
        if is_gray(img):
            img = cv2.cvtColor(img_in, cv2.COLOR_GRAY2BGR)
        img_y, img_x = img_in.shape[:2]
        for rho, theta in extra_in:
            pt1, pt2 = self._get_endpoints(rho, theta, img_x, img_y)
            img = cv2.line(
                img=img,
                pt1=pt1,
                pt2=pt2,
                color=self.color,
                thickness=self.thickness,
                lineType=self.line_type,
            )
            if self.show_points:
                img = cv2.circle(
                    img=img,
                    center=(round(rho * np.cos(theta)), round(rho * np.sin(theta))),
                    radius=3,
                    color=(255 - b, 255 - g, 255 - r),
                    thickness=3,
                )
        return img

    def _get_endpoints(self, rho, theta, img_x, img_y):
        """Return some endpoints for a line based on point and angle

        Kudos: https://stackoverflow.com/questions/18782873/houghlines-transform-in-opencv
        """
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = round(x0 + self.mult * (-b))
        y1 = round(y0 + self.mult * a)
        x2 = round(x0 - self.mult * (-b))
        y2 = round(y0 - self.mult * a)

        pt1 = (x1, y1)
        pt2 = (x2, y2)

        return pt1, pt2


class DrawLinesByEndpoints(BaseTransform):
    """Draws lines between points"""

    color = params.ColorPicker(default=(0, 0, 255))
    thickness = params.IntSlider(min_val=1, max_val=100, default=2)
    line_type = params.ComboBox(
        options=list(cvc.LINES.keys()), options_map=cvc.LINES, default="8-Connected"
    )

    def draw(self, img_in, extra_in):
        """Draws lines on img_in via point pairs

        Args:
            img_in (np.ndarray): Image in to draw onto
            extra_in ([(x1, y1, x2, y2), ..]): Iterable of two point pairs

        Returns:
            np.ndarray: Updated image
        """
        img = np.copy(img_in)
        if extra_in is None:
            return img

        if is_gray(img):
            img = cv2.cvtColor(img_in, cv2.COLOR_GRAY2BGR)
        for x1, y1, x2, y2 in extra_in:
            img = cv2.line(
                img=img,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=self.color,
                thickness=self.thickness,
                lineType=self.line_type,
            )
        return img


class DrawCircles(BaseTransform):
    """Draws circles"""

    color = params.ColorPicker(default=(0, 0, 255))
    thickness = params.IntSlider(min_val=1, max_val=100, default=2)
    line_type = params.ComboBox(
        options=list(cvc.LINES.keys()), options_map=cvc.LINES, default="8-Connected"
    )

    def draw(self, img_in, extra_in):
        """Draws circles onto img_in via x, y, and radius

        Args:
            img_in (np.ndarray): Incoming image to draw on
            extra_in ([(x, y, radius), ...]): Iterable of (x, y, radius) tuples

        Returns:
            np.ndarray: Updated Image
        """

        img = np.copy(img_in)
        if extra_in is None:
            return img

        if is_gray(img):
            img = cv2.cvtColor(img_in, cv2.COLOR_GRAY2BGR)

        for x, y, radius in extra_in:
            img = cv2.circle(
                img=img,
                center=(x, y),
                radius=int(radius),
                color=self.color,
                thickness=self.thickness,
                lineType=self.line_type,
            )
        return img


class DrawCirclesFromPoints(BaseTransform):
    """Draws circles from incoming Points"""

    color = params.ColorPicker(default=(0, 0, 255))
    radius = params.IntSlider(min_val=1, max_val=50, default=2)
    thickness = params.IntSlider(min_val=-1, max_val=20, default=-1)
    line_type = params.ComboBox(
        options=list(cvc.LINES.keys()), options_map=cvc.LINES, default="8-Connected"
    )

    def draw(self, img_in, extra_in):
        """Draws circles onto img_in via x, y

        Args:
            img_in (np.ndarray): Incoming image to draw on
            extra_in ([(x, y), ...]): Iterable of (x, y) tuples

        Some tools like GetGoodFeaturesToTrack return an N x 1 x 2 array, so
        if that's found we reshape it into N x 2 first.

        Returns:
            np.ndarray: Updated Image
        """

        if extra_in is None:
            return img_in, extra_in

        img = np.copy(img_in)

        if is_gray(img_in):
            img = cv2.cvtColor(img_in, cv2.COLOR_GRAY2BGR)

        points = extra_in

        if isinstance(extra_in, np.ndarray):
            if len(extra_in.shape) == 3:
                points = extra_in.reshape(extra_in.shape[0], 2)

        for x, y in points:
            img = cv2.circle(
                img=img,
                center=(x, y),
                radius=self.radius,
                color=self.color,
                thickness=self.thickness,
                lineType=self.line_type,
            )
        return img, extra_in


class DrawCornerSubPix(BaseTransform):
    """Draws points for CornerSubPix and its original contour points"""

    good_feat_color = params.ColorPicker(default=(0, 0, 255))
    good_feat_radius = params.IntSlider(min_val=1, max_val=50, default=3)
    good_feat_thickness = params.IntSlider(min_val=-1, max_val=20, default=-1)
    good_feat_line_type = params.ComboBox(
        options=list(cvc.LINES.keys()), options_map=cvc.LINES, default="8-Connected"
    )
    corners_color = params.ColorPicker(default=(0, 255, 0))
    corners_radius = params.IntSlider(min_val=1, max_val=50, default=1)
    corners_thickness = params.IntSlider(min_val=-1, max_val=20, default=-1)
    corners_line_type = params.ComboBox(
        options=list(cvc.LINES.keys()), options_map=cvc.LINES, default="8-Connected"
    )

    def draw(self, img_in, extra_in):
        """Draws circles onto img_in via x, y, and radius

        Args:
            img_in (np.ndarray): Incoming image to draw on
            extra_in ((getGoodFeatures, cornerSubPix)): Results of getGoodFeatures
                and CornerSubPix

        Some tools like GetGoodFeaturesToTrack return an N x 1 x 2 array, so
        if that's found we reshape it into N x 2 first.

        Returns:
            np.ndarray: Updated Image
        """

        if extra_in is None:
            return img_in, extra_in

        img = np.copy(img_in)

        if is_gray(img_in):
            img = cv2.cvtColor(img_in, cv2.COLOR_GRAY2BGR)

        orig_points = extra_in[0]
        orig_points = orig_points.reshape(orig_points.shape[0], 2)

        new_points = extra_in[1]
        new_points = new_points.reshape(new_points.shape[0], 2)

        for x, y in orig_points:
            img = cv2.circle(
                img=img,
                center=(x, y),
                radius=self.good_feat_radius,
                color=self.good_feat_color,
                thickness=self.good_feat_thickness,
                lineType=self.good_feat_line_type,
            )

        for x, y in new_points:
            img = cv2.circle(
                img=img,
                center=(x, y),
                radius=self.corners_radius,
                color=self.corners_color,
                thickness=self.corners_thickness,
                lineType=self.corners_line_type,
            )
        return img, extra_in


class ClusterGenerator(BaseTransform):
    """Generates clusters of points"""

    img_size = params.Dimensions2D(min_val=100, max_val=800, default=(250, 250))
    clusters = params.IntSlider(min_val=1, max_val=10, default=5)
    points_per_cluster = params.IntSlider(min_val=1, max_val=50, default=25)
    sigma = params.IntSlider(
        min_val=1,
        max_val=50,
        default=15,
        help_text="Sigma for gaussian spread from point centers",
    )

    def draw(self, img_in, extra_in):
        """Return original image and generated points"""
        points = self._generate_points()
        img = np.zeros((*self.img_size, 3), dtype=np.uint8)

        return img, points

    def _generate_points(self):
        """Return random set of x,y points in clusters"""
        # Create cluster centers
        y_max, x_max = self.img_size
        x_center = np.random.randint(1, x_max, size=(self.clusters, 1))
        y_center = np.random.randint(1, y_max, size=(self.clusters, 1))

        # distance from center points
        radii = np.random.normal(
            1, self.sigma, size=(self.clusters, self.points_per_cluster)
        )
        angles = 2 * np.pi * np.random.random((self.clusters, self.points_per_cluster))

        # New coordinates
        x = radii * np.cos(angles) + x_center
        y = radii * np.sin(angles) + y_center

        # Reshape into 2d array of points
        x = x.reshape((np.product(x.shape), 1))
        x = x.clip(0, x_max)
        y = y.reshape((np.product(y.shape), 1))
        y = y.clip(0, y_max)
        points = np.hstack([x, y]).astype(np.int)
        return points


class DrawKMeansPoints(BaseTransform):
    """Draws points for Kmeans"""

    point_size = params.IntSlider(min_val=1, max_val=25, default=2)
    center_point_size = params.IntSlider(min_val=1, max_val=25, default=4)
    center_color = params.ColorPicker(default=(0, 255, 0))
    h = params.IntSlider(min_val=0, max_val=179, default=100,)
    s = params.IntSlider(min_val=1, max_val=255, default=255)
    v = params.IntSlider(min_val=1, max_val=255, default=255)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_clusters = 0
        self.colors = None

    def draw(self, img_in, extra_in):
        img = np.copy(img_in)
        points, labels, centers = extra_in
        centers = centers.astype(np.int)
        self.n_clusters = len(centers)

        # Reset colors - no this is not efficient
        colors = self._get_colors(self.h, self.v, self.s, self.n_clusters)

        # Draw points by color
        for idx, point in enumerate(points):
            img = cv2.circle(
                img=img,
                center=tuple(point),
                radius=self.point_size,
                color=tuple(colors[labels[idx][0]]),
                thickness=cv2.FILLED,
                lineType=cv2.LINE_AA,
            )

        # Draw point centers
        for point in centers:
            img = cv2.circle(
                img=img,
                center=tuple(point),
                radius=self.center_point_size,
                color=self.center_color,
                thickness=cv2.FILLED,
                lineType=cv2.LINE_AA,
            )

        return img

    def _get_colors(self, h, s, v, n_colors):
        """Draw colors around the Hue at given s and v"""
        # cv2 hue goes from 0-179
        offset = 305 / 2 / n_colors
        colors = []
        for _ in range(n_colors):
            color = np.array([[[h, s, v]]], dtype=np.uint8)
            bgr_color = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)
            # color _must_ be an regular int; not an nptype
            colors.append(tuple(bgr_color[0][0].tolist()))
            h += offset
            if h > 180:
                h %= offset
        return colors


class BitwiseAnd(BaseTransform):
    """Perform a cv2.bitwise_and"""

    def draw(self, img_in, extra_in):
        """Return bitwise_and using extra_in as mask"""
        return cv2.bitwise_and(img_in, img_in, mask=extra_in)


class CvtColor(BaseTransform):
    """Convert from BGR to a different color space"""

    color_range = params.ComboBox(
        options=["BGR", "HSV", "LAB", "YUV"], options_map=cvc.COLOR_BGR2
    )

    def draw(self, img_in, extra_in):
        # NOTE: we always pass the original image on as extra_in
        # so it can be displayed later

        # We have to assume incoming image is BGR, so don't want update if so
        if self.color_range != "BGR":
            return cv2.cvtColor(img_in, self.color_range), img_in
        return img_in, img_in


class DisplayHarris(BaseTransform):
    """Displays top N points of Harris Corners"""

    top_n_points = params.IntSlider(
        min_val=1, max_val=1000, default=100, help_text="Show top N points"
    )
    point_color = params.ColorPicker(default=(0, 0, 255))

    def draw(self, img_in, extra_in):
        """Display harris corner points overlayed on img_in

        Args:
            img_in (np.ndarray [uint8]): Single channel input image
            extra_in (np.ndarray [float]): Output from cornerHarris
        """
        top_points = self._get_top_points(extra_in, self.top_n_points)
        img = img_in
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for y, x in zip(top_points[0], top_points[1]):
            img = cv2.circle(
                img=img, center=(x, y), radius=2, color=self.point_color, thickness=2
            )
        return img

    def _get_top_points(self, values, n_points):
        """Return index of top `n_points` from `values`"""
        sorted_vals = np.argsort(values.ravel())[::-1][:n_points]
        return np.unravel_index(sorted_vals, values.shape)


class BlankCanvas(BaseTransform):
    """Create a blank canvas"""

    img_shape = params.Dimensions2D(
        min_val=100, max_val=800, default=(250, 250), read_only=True
    )

    def draw(self, img_in, extra_in):
        """Return a black canvas of size self.img_shape"""
        shape = self.img_shape + (3,)
        return np.zeros(shape, dtype=np.uint8)


class DrawContours(BaseTransform):
    color = params.ColorPicker(default=(0, 255, 0))
    contour_index = params.IntSlider(
        min_val=-1,
        max_val=1,
        step=1,
        default=-1,
        help_text="-1 -> All Contours, otherwise only draw specific contour",
    )
    thickness = params.IntSlider(
        min_val=-1,
        max_val=10,
        default=2,
        help_text="-1 -> Filled; otherwise draw lines at specified thickness",
    )
    line_type = params.ComboBox(
        options=["4-Connected", "8-Connected", "Anti Aliased"],
        options_map=cvc.LINES,
        default="8-Connected",
    )

    def draw(self, img_in, contours):
        """Draw's contours

        Args:
            img_in (np.ndarray): image to draw on
            contours (np.ndarray): Output from ``findContours``

        Returns:
            np.ndarray, np.ndarray: Updated Image, Original Contours
        """
        # Update the max number of contours based
        self._contour_index.set_max(len(contours))

        cv2.drawContours(
            image=img_in,
            contours=contours,
            contourIdx=self.contour_index,
            color=self.color,
            thickness=self.thickness,
            lineType=self.line_type,
        )
        return img_in, contours


class DrawGaussianKernel(BaseTransform):
    """Display Transform for GetGaussianKernel"""

    def draw(self, img_in, extra_in):
        """extra_in should be the output from cv2.getGaussianKernel

        This assumes the preceding transform is GetGaussianKernel
        """
        gaus = self.get_transform(1)
        img = np.tile(extra_in, (1, gaus.k_size)) * 255
        num_rows = img_in.shape[0]
        scale = num_rows / gaus.k_size
        img = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale)
        return img


class DrawPyrDown(BaseTransform):
    """Display Transform for PyrDown"""

    def draw(self, img_in, extra_in):
        """extra_in should be a list of the new images from pyrDown"""
        images = extra_in

        depth = 0 if len(images[0].shape) == 2 else images[0].shape[2]

        col_pix = [0] + [x.shape[1] for x in images]
        col_tot = np.sum(col_pix)
        row_tot = images[0].shape[0]
        shape = (row_tot, col_tot, depth) if depth else (row_tot, col_tot)
        out = np.zeros(shape, dtype=np.uint8)
        col_starts = np.cumsum(col_pix)

        for idx in range(len(images)):
            col_s = col_starts[idx]
            cr, cc = images[idx].shape[:2]
            out[0:cr, col_s : col_s + cc] = images[idx]
        return out
