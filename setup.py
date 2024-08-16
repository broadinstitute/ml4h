from setuptools import setup, find_packages


# Function to read requirements from a file
def load_requirements(filename):
    with open(filename, 'r') as file:
        return file.read().splitlines()

    
setup(
    name='ml4h',
    version='0.0.15rc3',
    description='Machine Learning for Health python package',
    #long_description= 'README.md'.read_text(encoding='utf-8'),  # Optional
    long_description_content_type='text/markdown',
    url='https://github.com/broadinstitute/ml4h',
    python_requires='>=3.6',
    #install_requires=["ml4ht", "tensorflow", "pytest", "numcodecs"], # requirements
    install_requires=load_requirements('docker/vm_boot_images/config/tensorflow-requirements.txt'),  # requirements
    #install_requires=requirements,
    packages=find_packages(),
)
