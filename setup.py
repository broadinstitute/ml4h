import os
from setuptools import setup, find_packages


this_directory = os.path.abspath(os.path.dirname(__file__))
requirements_path = os.path.join(this_directory, 'docker/vm_boot_images/config/tensorflow-requirements.txt')
# Function to read requirements from a file


def load_requirements(filename):
    with open(filename, 'r') as file:
        return file.read().splitlines()


setup(
    name='ml4h',
    version='0.0.17',
    description='Machine Learning for Health python package',
    long_description=os.path.join(this_directory, 'README.md').read_text(encoding='utf-8'),  # Optional
    long_description_content_type='text/markdown',
    url='https://github.com/broadinstitute/ml4h',
    python_requires='>=3.6',
    #install_requires=["ml4ht", "tensorflow", "pytest", "numcodecs"], # requirements
    install_requires=load_requirements(requirements_path),  # requirements
    #install_requires=requirements,
    packages=find_packages(),
)
