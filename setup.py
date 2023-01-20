import pathlib
from setuptools import setup, find_packages

here = pathlib.Path(__file__).parent.resolve()
# Get the requirements from the requirements file
# requirements = 'docker/vm_boot_images/config/tensorflow-requirements.txt'.read_text(encoding='utf-8')
long_description = (here / 'README.md').read_text(encoding='utf-8')
setup(
    name='ml4h',
    version='0.0.3dev3',
    description='Machine Learning for Health python package',
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',
    url='https://github.com/broadinstitute/ml4h',
    python_requires='>=3.6',
    install_requires=["ml4ht"],
    packages=find_packages(),
)
