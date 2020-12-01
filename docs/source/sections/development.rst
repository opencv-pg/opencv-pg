Development
===========
To install in development mode::

    git clone https://github.com/opencv-pg/opencv-pg
    pip install -e opencv-pg/[dev]


Running Tests
^^^^^^^^^^^^^

.. code-block:: bash

    cd tests
    pytest


Generating Docs
^^^^^^^^^^^^^^^
.. code-block:: bash

    # Top level docs directory
    cd docs
    sphinx-apidoc -f -o source/ ../src/opencv_pg
    make html`

Output will be in the ``docs/build/html/`` directory.
