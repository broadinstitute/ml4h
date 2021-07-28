import pathlib
from setuptools import setup, find_packages

here = pathlib.Path(__file__).parent.resolve()
# Get the requirements from the requirements file
requirements = (here / 'docker/vm_boot_images/config/tensorflow-requirements.txt').read_text(encoding='utf-8')
long_description = (here / 'README.md').read_text(encoding='utf-8')
setup(
    name='ml4h',
    version='0.0.1.dev6',
    description='Machine Learning for Health python package',
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',
    url='https://github.com/broadinstitute/ml4h',
    python_requires='>=3.6',
    install_requires=requirements + \
                     "ml4ht @ git+https://github.com/broadinstitute/torch_ml4h",
    packages=find_packages(),
)
