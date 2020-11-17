Overview
========
The OpenCV Playground is composed of two major components:

#. ``Playground``
#. ``Custom Pipeline Launcher``

Playground
----------
The Playground is a Qt5 application that brings together improved documentation alongside OpenCV functions with the ability to explore the effects of function parameters on an image in real time.

It comes pre-populated with a list of built-in OpenCV functions. When a function is selected, the documentation is displayed for that function and the parameters can be manipulated via various Qt5 widgets. The results of the parameter changes are reflected in the image. See :ref:`Usage` for invocation.

Interactive Pipeline
------------------------
The package contains the :func:`launch_pipeline<opencv_pg.pipeline_launcher.launch_pipeline>` function that can be used to launch your own custom Pipeline. It takes a :class:`Pipeline<opencv_pg.models.pipeline.Pipeline>` instance and then displays all of your ``Windows`` so you can interact with your ``Transforms`` and view the output. More information is available in the :ref:`Pipeline Launcher` section.
