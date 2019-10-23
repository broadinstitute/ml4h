from setuptools import setup, find_packages

setup(
    name='ml4cvd',
    version='0.0.1',
    description='Machine Learning for CardioVascular Disease package',
    url='https://github.com/broadinstitute/ml',
    python_requires='>=3.6',
    install_requires=['tensorflow', 'keras'],
    packages=find_packages(),
)
