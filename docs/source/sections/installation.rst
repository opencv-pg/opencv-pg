Installation
============

Requirements
------------
This has been tested using the following:

* Python: ``3.7.4``
* opencv-contrib-python-headless: ``4.4.0.46``

Quick Installation
------------------
Install the package using ``pip``::

    pip install opencv-pg


Or clone the package and install::

    git clone https://github.com/opencv-pg/opencv-pg
    pip install opencv-pg/


Ubuntu Users
^^^^^^^^^^^^
On Ubuntu 16.04 (others currently untested), there may be missing links to ``xcb`` related shared objects.

tldr;
::

    sudo apt-get install --reinstall libxcb-xinerama0

-----------

If you see errors about ``xcb``, you can perform the following to help troubleshoot. In your terminal, make the following export::

    export QT_DEBUG_PLUGINS=1

Run ``opencvpg`` again and validate the output. The final lines will likely mention details about files not found. Likely ``libxcb-xinerama.so.0``.

Run the following::

    cd your_venv/lib/pythonX.X/site-packages/PySide2/Qt/plugins/platforms/
    ldd libqxcb.so | grep "not found"

This will print any missing links. In our case, ``libxcb-xinerama.so.0`` showed up a couple times. Reinstalling the package as follows resolved the issue::

    sudo apt-get install --reinstall libxcb-xinerama0

Once it’s working, you may want to disable that ``QT_DEBUG_PLUGINS`` env variable so it doesn’t throw extra garbage in your output.


Development Installation
------------------------
To install for development::

    git clone https://github.com/opencv-pg/opencv-pg
    pip install -e opencv-pg/[dev]


Running Tests
^^^^^^^^^^^^^
::

    cd tests
    pytest


Building Docs
^^^^^^^^^^^^^
We use sphinx for documentation management::

    # Top level docs directory
    cd docs
    sphinx-apidoc -f -o source/ ../src/opencv_pg
    make html