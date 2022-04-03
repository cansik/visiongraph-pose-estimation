from setuptools import setup, find_packages

NAME = 'vgpose'

required_packages = find_packages()

with open('requirements.txt') as f:
    required = [line for line in f.read().splitlines() if not line.startswith("-")]

setup(
    name=NAME,
    version='1.0.0',
    packages=required_packages,
    entry_points={
        'console_scripts': [
            'vgpose = vgpose.__main__:main',
        ],
    },
    url='https://github.com/cansik/visiongraph-pose-estimator',
    license='MIT License',
    author='Florian Bruggisser',
    author_email='github@broox.ch',
    description='A visiongraph based pose estimator example with performance profiling.',
    install_requires=required,
)
