import logging
import traceback

import numpy as np
import copy

from qtpy import QtCore

from .params import Param

log = logging.getLogger(__name__)


def _break_result_into_parts(result):
    """Returns 2-tuple of (np.ndarray, None) or (np.ndarray, Object)

    Args:
        result (ndarray or tuple/list): image or (image, object)

    Raises:
        TypeError:

    Returns:
        tuple: (np.ndarray, Object or None)
    """
    if isinstance(result, (tuple, list)):
        if len(result) == 1:
            return result[0], None
        if len(result) == 2:
            return result[0], result[1]
        else:
            raise TypeError(
                "Transform return must be: np.ndarray or (np.ndarray, ), or "
                "(np.ndarray, Object). Got %s",
                result,
            )
    return result, None


class DeclarativeFieldBase(type):
    """Base MetaClass for setting up fields and making Param's properties

    These will specifically apply only to Param objects.

    The goal here is to allow a Transform to declare `Param` class variables
    that are then converted into instance level parameters in the instance.
    """

    def __new__(cls, name, bases, attrs):
        """Auto Creates properties for Param class attributes

        Concept taken from here: https://gist.github.com/KrzysztofCiba/3813797
        """

        # Create accessor methods that will be applied to each class
        # Will be accessing the _value prop of the underlying Param
        def _getter(self, name):
            return getattr(getattr(self, f"_{name}"), "_value")

        def _setter(self, name, value):
            setattr(getattr(self, f"_{name}"), "_value", value)

        def _make_prop(self, name, value):
            """Creates instance `_name` attr and `name` property for it"""
            _name = f"_{name}"

            getter = lambda self: self._getter(name)  # noqa: E731
            setter = lambda self, _value: self._setter(name, _value)  # noqa: E731

            setattr(self, _name, copy.deepcopy(value))
            # Add accessors to class
            setattr(self.__class__, f"{name}", property(getter, setter))

        params = []
        accessors = {"_getter": _getter, "_setter": _setter, "_make_prop": _make_prop}
        for key, value in attrs.items():
            if isinstance(value, Param):
                params.append((key, value))
        attrs["_params"] = params
        attrs.update(accessors)
        return super().__new__(cls, name, bases, attrs)


class BaseTransform(metaclass=DeclarativeFieldBase):
    """Base class for all Transforms

    Args:
        doc_filename (str, optional): Name of the documentation file located
            in src/docs/source_docs/ if different than the current
            <ClassName>.html
    """

    # Will use __name__.html if not set
    doc_filename = None

    def __init__(self):
        super().__init__()
        self.params = []
        self.error = None
        self.index = None
        self.window = None
        self.last_in = None
        self.extra_in = None
        self.enabled = True

        for name, value in self._params:
            # Create all the fields and accessors
            self._make_prop(name, value)
            obj = getattr(self, f"_{name}")
            obj._transform = self
            if obj.label is None:
                obj.label = name.replace("_", " ").title()
            self.params.append(obj)

    @classmethod
    def get_doc_filename(cls):
        """Return the classes doc filename"""
        if cls.doc_filename is None:
            return f"{cls.__name__}.html"
        return cls.doc_filename

    def start_pipeline(self):
        """Starts the pipeline from this transform"""
        self.window.start_pipeline(self.index)

    def get_transform(self, index):
        """Returns Transform at ``index`` in ``self.window``

        Args:
            index (int): Index of Transform to fetch

        Returns:
            BaseTransform: Transform instance
        """
        return self.window.transforms[index]

    def get_info_widget(self):
        """Optionally return a widget that provides info about the transform.

        If this is filled out, it will be located above the params of this
        transform's params group box.
        """
        pass

    def interconnect_widgets(self):
        """Can be used to connect signals/slots between widgets

        This is called after all widgets have been instantiated for each Param.
        The various params can be accessed via ``self.params``.
        """
        pass

    def _draw(self, img_in, extra_in):
        """Performs the transform, possibly storing the inputs for later use

        Args:
            img (np.array): Image to operate on
            extra_in (object, None): Can be anything that needs to be passed
                down from the previous transform such as calculation results.
                Must be able to ``deepcopy`` it.

        Returns:
            (np.ndarray, object), : Output image, any object
        """

        # Starting the pipeline here; inputs are None, so use last stored
        if img_in is None or len(img_in.shape) == 0:
            img_out = np.copy(self.last_in)
            extra_out = copy.deepcopy(self.extra_in)
        # We were passed something so store it
        else:
            self.last_in = np.copy(img_in)
            self.extra_in = copy.deepcopy(extra_in)
            img_out = np.copy(img_in)
            extra_out = copy.deepcopy(extra_in)

        # Bypass since disabled
        if not self.enabled:
            return img_out, extra_out

        # Run transform; on error return the inputs
        try:
            self.update_widgets_state()
            out = self.draw(img_out, extra_out)
            img_out, extra_out = _break_result_into_parts(out)
            self.error = None
        except Exception as e:
            log.exception(e)
            self.error = traceback.format_exc()

        return img_out, extra_out

    @QtCore.Slot(bool)
    def handle_enabled_changed(self, enabled):
        """Sets whether transform should be enabled then reruns pipeline

        Args:
            enabled (Bool): True if enabled
        """
        self.enabled = enabled
        self.start_pipeline()

    def draw(self, img_in, extra_in):
        """Override performing your transformation

        Must return either:
            - np.ndarray - an image
            - (np.ndarray, object) - an image and any additional object you want
                passed onto the next transform. Must be able to ``deepcopy`` it.

        Args:
            img_in (np.ndarray): Copy of image from previous transform
            extra_in (object): Any object that can be deepcopied

        Returns:
            np.ndarray, object (optional): Your modified image or optionally
                your modified image and some additional object you want to pass
                onto the next transform (as a tuple), eg: (np.ndarray, object)
        """
        raise NotImplementedError

    def update_widgets_state(self):
        """Override to update the state of the widgets within this transform

        This can be used to conditionally change/activate/decativate one widget
        based on the state of another widget. This is called just prior to the
        ``draw`` method.

        # NOTE: Might be able to decouple these Transforms and widgets more
        # by emitting signals from the transform and connecting them to the
        # widgets?
        # NOTE: This requires making BaseTransform a subclass of QObject.
        # This comes with its own complications as you now can't deepcopy it
        # without specfying how.
        """
        pass
