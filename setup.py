"""Setup file."""
from setuptools import setup, find_packages
from setuptools.command.install import install

def readme():
    with open('README.rst') as f:
        return f.read()
   
setup(  name='motrack',
        version='0.1',
        description='Object tracking for Biologists',
        long_description=readme(),
        keywords=['motion tracking', 'laboratory', 'mouse'],
        classifiers = [],
        url='https://github.com/EIN-lab/motion-tracking',
        download_url='https://github.com/EIN-lab/motion-tracking/archive/master.zip',
        author='Martin Holub',
        author_email='mholub.ethz@gmail.com',
        license='MIT',
        packages=find_packages(),
        setup_requires=['pytest-runner'],
        tests_require=['pytest'],
        include_package_data=True,
        zip_safe=False)