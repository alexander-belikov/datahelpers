import os
from setuptools import setup
from distutils.core import setup
import numpy as np

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="datahelpers",
    version="0.0.2",
    author="Alexander Belikov",
    author_email="abelikov@gmail.com",
    description="tools to wrangle data",
    license="BSD",
    keywords="data wrangle pandas mysql",
    url="git@github.com:alexander-belikov/datahelpers.git",
    packages=['datahelpers'],
    long_description=read('README'),
    classifiers=[
        "Development Status :: 0 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
    install_requires=[
        'numpy',
        'pymysql', 'pandas',
        'setuptools', 'PyWavelets',
        'matplotlib', 'seaborn',
        'pathos', 'Levenstein'],
    include_dirs=[np.get_include()]
)
