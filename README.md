# OpenCV Playground
The OpenCV Playground is a Qt6 application that brings together improved documentation alongside OpenCV functions with the ability to explore the effects of function parameters on an image in real time.

It also comes with a custom `Pipeline Launcher` that allows you to build and interact with your own sequence of image transformations along with custom build functions.

Full documentation can be found on [Read the Docs](https://opencv-pg.readthedocs.io/en/latest/).

## Demo
<a href="https://drive.google.com/uc?export=view&id=1i4jmCHebu1_ognIwj2n4vtpCaT8BHWGI"><img src="https://media.giphy.com/media/GQj3aod8oKoxpJ4sC3/giphy.gif" style="width: 500px; height: auto;" /></a>


## Installation
Currently tested with python 3.8.10/3.10.0 and opencv-contrib-python-headless 4.8.0.76 on an M1 Mac.

From PyPi:

```shell
pip install opencv-pg
```

From Github Repo:

```shell
pip install git+https://github.com/opencv-pg/opencv-pg
```

### Note for Linux Users
**NOTE:** I no longer have access to anything but a Mac (as of `1.0.3`), so I can't confirm if the below still stands. It did in `1.0.2`.

On Ubuntu 16.04 (others currently untested), there may be missing links to xcb related shared objects.

tldr;

```shell
sudo apt-get install --reinstall libxcb-xinerama0
```
------

**Digging Deeper**

If you see errors about xcb, you can perform the following to help troubleshoot. In your terminal, make the following export:

```shell
export QT_DEBUG_PLUGINS=1
```

Run opencvpg again and validate the output. The final lines will likely mention details about files not found. Likely `libxcb-xinerama.so.0`.

Run the following:

```shell
cd your_venv/lib/pythonX.X/site-packages/PySide2/Qt/plugins/platforms/
ldd libqxcb.so | grep "not found"
```

This will print any missing links. In our case, `libxcb-xinerama.so.0` showed up a couple times. Reinstalling the package as follows resolved the issue:

```shell
sudo apt-get install --reinstall libxcb-xinerama0
```

Once it’s working, you may want to disable that `QT_DEBUG_PLUGINS` env variable so it doesn’t throw extra garbage in your output.

## Usage
### Playground
To launch the OpenCV Playground with:
* The built-in image:

```shell
opencvpg
```

* An image of your choice:

```shell
opencvpg --image <path-to-image.png>
```

* Without the documentation window:

```shell
opencvpg --no-docs
```

### Custom Pipeline
The following is an example of building a custom Pipeline.

```python
from opencv_pg import Pipeline, Window, launch_pipeline
from opencv_pg import support_transforms as supt
from opencv_pg import transforms as tf

if __name__ == '__main__':
    my_image = '/path/to/image.png'

    # Creates two windows
    pipeline = Pipeline([
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
```

Then run the file.

## Development
### Installation
To install in development mode:

```shell
git clone https://github.com/opencv-pg/opencv-pg
pip install -e opencv-pg/[dev]
```

### Running Tests
```shell
cd tests
pytest
```

### Generating Docs
* Go into the top level `docs` directory
* run `sphinx-apidoc -f -o source/ ../src/opencv_pg`
* run `make html`

Output will be in the `docs/build/html/` directory.
