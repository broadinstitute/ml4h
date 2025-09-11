import pathlib
from setuptools import setup, find_packages

here = pathlib.Path(__file__).parent.resolve()
# Get the requirements from the requirements file
requirements = (here / 'docker/vm_boot_images/config/tensorflow-requirements.txt').read_text(encoding='utf-8')
long_description = (here / 'README.md').read_text(encoding='utf-8')
setup(
    name='ml4h',
    version='v0.1.1',
    description='Machine Learning for Health python package',
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',
    url='https://github.com/broadinstitute/ml4h',
    python_requires='>=3.6',
    #install_requires=["ml4ht", "tensorflow"], # requirements
    install_requires=requirements,
    packages=find_packages(),
)
