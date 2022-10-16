"""
high level support for doing this and that.
"""
import os
from setuptools import find_packages, setup

def read(fname):
    """A dummy docstring."""
    return open(os.path.join(os.path.dirname(__file__), fname),encoding="utf-8").read()


setup(
    name='SchNet',
    version='0.1.1',
    author='Kristof T. Schütt',
    email='kristof.schuett@tu-berlin.de',
    url='https://github.com/atomistic-machine-learning/SchNet',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    license='MIT',
    description='SchNet - a deep learning architecture for quantum chemistry',
    long_description=read('README.md')
)
