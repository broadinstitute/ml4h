import pathlib
from setuptools import setup, find_packages

here = pathlib.Path(__file__).parent.resolve()
setup(
    name='ml4h',
    version='0.0.2',
    description='Machine Learning for Health python package',
    url='https://github.com/broadinstitute/ml4h',
    python_requires='>=3.6',
    packages=find_packages(),
)
