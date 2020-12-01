#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()


setup(
    name='opencv-pg',
    version='1.0.0',
    license='GPL-3.0-or-later',
    description='Qt5 GUI Application for realtime exploration of OpenCV functions',
    long_description="%s\n%s" % (
        re.compile('^.. start-badges.*^.. end-badges', re.M | re.S).sub('', read('README.md')),
        re.sub(':[a-z]+:`~?(.*?)`', r'``\1``', read('CHANGELOG.md'))
    ),
    long_description_content_type="text/markdown",
    author='',
    author_email='',
    url='https://github.com/opencv-pg/opencv-pg',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    project_urls={
        'Changelog': 'https://github.com/opencv-pg/opencv-pg/blob/master/CHANGELOG.md',
        'Issue Tracker': 'https://github.com/opencv-pg/opencv-pg/issues',
    },
    keywords=[
        'opencv', 'cv2', 'cv', 'computer vision',
    ],
    python_requires='>=3.6.*',
    install_requires=[
        'pyside2==5.15.0',
        'qtpy==1.9.0',
        'opencv-contrib-python-headless==4.4.0.46',
        'jinja2==2.11.2'
    ],
    extras_require={
        'dev': [
            'pytest==6.0.1',
            'flake8',
            'sphinx==3.2.1',
        ]
    },
    setup_requires=[
        'pytest-runner',
    ],
    entry_points={
        'console_scripts': [
            'opencvpg = opencv_pg.launcher:main',
            'opencvpg_docview = opencv_pg.launcher:docview'
        ]
    },
)