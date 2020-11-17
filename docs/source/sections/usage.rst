Usage
=====

Launching
---------
Once installed, the application can be launched as follows:

* With the default image::

    opencvpg

* With a specific image::

    opencvpg --image <path to image>

* Without the documenation window::

    opencvpg --no-docs


Using
-----
Some tips about interacting with the application:

* In many cases, the slider min/max limits can be changed by double clicking on them and entering a new value.
    * This does mean you could enter invalid values.
* In some cases, you can hover over the parameter label for additional context/help.
* The image can be zoomed (scroll wheel), panned (left-click drag), and the view reset via a right-click context menu.
* If an ``Exception`` occurs while running the OpenCV method (likely due to invalid paramters), the input image will be passed back out as the output and the exception will be logged in the terminal.
