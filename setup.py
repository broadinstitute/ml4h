from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='ml4cvd',
    version='0.0.1',
    description='Machine Learning for Disease',
    url='https://github.com/broadinstitute/ml',
    python_requires='>=3.6',
    install_requires=requirements,
    packages=find_packages(),
)
