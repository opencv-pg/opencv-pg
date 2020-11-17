# OpenCV Playground
The OpenCV Playground is a Qt5 application that brings together improved documentation alongside OpenCV functions with the ability to explore the effects of function parameters on an image in real time.

It also comes with a custom `Pipeline Launcher` that allows you to build and interact with your own sequence of image transformations.

**TODO: Update Read The Docs link - not yet active**
Full documentation can be found on [Read the Docs](https://opencvpg.readthedocs.io/en/latest/).

## Installation
Currently tested with python 3.7.4 and opencv-headless-4.4.0.46

From PyPi (not yet active):

```shell
pip install opencv-pg
```

From Github Repo:

```shell
pip install git+https://github.com/opencv-pg/opencv-pg
```

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
cd opencv-pg
pip install -e ./[dev]
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