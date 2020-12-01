Pipeline Backend
================
The ``Pipeline`` is the core backend of the ``Playground``. It is responsible for running a sequence of ``Transforms`` and displaying the output. It can also be run in conjunction with the :ref:`Pipeline Launcher`. It consists of the following components:

* :class:`Pipeline<opencv_pg.models.pipeline.Pipeline>`
* :class:`Windows<opencv_pg.models.window.Window>`
* :class:`Transforms<opencv_pg.models.base_transform.BaseTransform>`
* :class:`Params<opencv_pg.models.params.Param>`

The following relationships are defined between these objects::

    Pipeline
    |- Window
    |   |- Transform
    |   |   |- Param
    |   |   |- ...
    |   |- ...
    |- ...

The components above will be described from the lowest level to the highest.

Param
-----
A :class:`Param<opencv_pg.models.params.Param>` is an input to the :class:`Transform<opencv_pg.models.base_transform.BaseTransform>` that can be manipulated via some kind of Qt Widget, such a slider, input box, select box, etc.

See the :mod:`Params<opencv_pg.models.params>` module for existing ``Param`` classes that can be used.

Defining a New Param
^^^^^^^^^^^^^^^^^^^^
A new :class:`Param<opencv_pg.models.params.Param>` can be defined as follows::

    class MyParam(Param):
        def __init__(self, my_arg, default=None, label=None, read_only=False, help_text=''):
            # You must call the super first
            super.__init__(default=default, label=label, read_only=read_only,
                           help_text=help_text):
            # define and store any new arguments
            self.my_arg = my_arg
            ...

        def _get_widget(self, parent=None):
            """Return a widget instance for this Param type"""
            # 1. Create and configure widget instance, passing the parent in
            # 2. Connect the widgets value changed handler to a function on this Param
            # 3. Return the widget instance

        @QtCore.Slot(<appropriate_type_here>)
        def _handle_value_changed(self, value):
            """Stores the updated value and runs the pipeline"""
            # NOTE: Do any necessary conversion to value here before storing it
            self._store_value_and_start(value)

A `nice introduction <https://wiki.qt.io/Qt_for_Python_Signals_and_Slots>`_ to the Signals and Slots mechanisms in Qt5.

Transform
---------
A :class:`Transform<opencv_pg.models.base_transform.BaseTransform>` defines either a manipulation that will be done on an image or can perform a calculation and pass the results onto the following :class:`Transforms<opencv_pg.models.base_transform.BaseTransform>`. We define class level :class:`Param<opencv_pg.models.params.Param>` variables to allow the user to interact with the :class:`Transforms<opencv_pg.models.base_transform.BaseTransform>`.

The :class:`Transform<opencv_pg.models.base_transform.BaseTransform>` defines a :func:`draw(img_in, extra_in)<opencv_pg.models.base_transform.BaseTransform.draw>` method that takes an image in and possibly extra information, and then returns an image or an image and some extra information as a ``tuple``. These are passed onto the next :class:`Transform<opencv_pg.models.base_transform.BaseTransform>`.

Each :class:`Param<opencv_pg.models.params.Param>` value can be accessed and set via ``self.param_name``. The actual :class:`Param<opencv_pg.models.params.Param>` instance is stored as ``self._param_name``.

Please see the :mod:`opencv_pg.models.transforms` and  :mod:`opencv_pg.models.support_transforms` modules for existing :class:`Transforms<opencv_pg.models.base_transform.BaseTransform>` which may be useful.

Creating a New Transform
^^^^^^^^^^^^^^^^^^^^^^^^
Creating your own ``Transform`` is easy!

A new :class:`Transform<opencv_pg.models.base_transform.BaseTransform>` can be defined as follows::

    from opencv_pg import BaseTransform
    from opencv_pg import params

    OPTIONS = {
        'Display 1': 'value1',
        'Display 2': 'value2',
    }

    class MyTransform(BaseTransform):
        param1_slider = params.IntSlider(min_val=0, max_val=255, step=1, default=100)
        combo = params.ComboBox(
            options=value_map.keys(), default='Display 1', options_map=OPTIONS
        ])
        checkbox = params.CheckBox()

        def draw(self, img_in, extra_in):
            """Required - must return an ndarray, or an (ndarray, object)"""
            out = cv2.some_function(
                img=img_in,
                param1=self.param1_slider,
                param2=self.combo,
                chk=self.checkbox
            )

            return out
            # or return out, something_extra

        def get_info_widget(self):
            """Optional: Return a QWidget that will be displayed as extra
            information above the Transform group"""
            pass

        def update_widgets_state(self):
            """Optional: update the state of other various widgets within this
            Transform, based on each other's state. Can be used to test one
            widget for a value, and enable/disable other widgets
            """
            pass

This ``Transform`` would display an Integer slider, a ComboBox and a CheckBox.

Window
------
A :class:`Window<opencv_pg.models.window.Window>` is composed of one or more :class:`Transforms<opencv_pg.models.base_transform.BaseTransform>`. Each :class:`Window<opencv_pg.models.window.Window>` is responsible for displaying the output of the last :class:`Transform<opencv_pg.models.base_transform.BaseTransform>` in its list, and then passing that output onto the first :class:`Transform<opencv_pg.models.base_transform.BaseTransform>` of the next :class:`Window<opencv_pg.models.window.Window>`.

Creating a Window
^^^^^^^^^^^^^^^^^
A window can be created as follows::

    window = Window([
        Transform1(),
        Transform2()
    ])

You can optionally pass a ``name`` argument to the ``Window`` to give it a meaningful window title. If no ``name`` is passed, it will be named ``Step N``, according to its position in the ``Pipeline``.

Pipeline
--------
The :class:`Pipeline<opencv_pg.models.pipeline.Pipeline>` represents the top level feature of the hierarchy. It sets up the windows and is responsible for running all the ``Transforms`` in the pipeline.

Creating a Pipeline
^^^^^^^^^^^^^^^^^^^
A Pipeline can be created in any of the following ways::

    # There is a single Transform
    pipeline1 = Pipeline(Transform())

    # If there are multiple Transforms, but only one Window
    pipeline2 = Pipeline([Transform1(), Transform2(), Transform3()])

    # If there are multiple Windows
    pipeline3 = Pipeline([
        Window([
            Transform1(),
            Transform2(),
        ]),
        Window([
            Transform2(),
            Transform3(),
        ])
    ])

Pipeline Launcher
-----------------
Now that you've created your own custom ``Params`` and ``Transforms``, we can put them all together into your own pipeline.

A custom ``Pipeline`` can be launched by your own code using the :func:`launch_pipeline<opencv_pg.pipeline_launcher.launch_pipeline>` function. When this is done, a Qt Window will be displayed for each ``Window`` in your ``Pipeline``.

Example::

    from opencv_pg import Pipeline, Window, launch_pipeline
    from opencv_pg import support_transforms as supt
    from opencv_pg import transforms as tf

    if __name__ == '__main__':
        my_image = '/path/to/image.png'

        pipeline = Pipeline([
            # You could also import and use your own Transforms
            Window([
                supt.LoadImage(my_image),
                supt.CvtColor(),
                tf.InRange(),
                supt.BitwiseAnd(),
            ]),
            Window([
                tf.Canny(),
            ]),
        ])

        launch_pipeline(pipeline)

This will show two ``Windows``. The first with the final output of the ``BitwiseAnd`` and the second with the output of the ``Canny`` operation.
