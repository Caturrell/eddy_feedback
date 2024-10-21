from setuptools import setup, find_packages

setup(
    name='eddy_feedback',  # Name of your package
    version='0.1',
    packages=find_packages(),  # Automatically find sub-packages like 'functions'
    install_requires=[],  # List any dependencies your package needs
    author='Your Name',
    description='A package for eddy feedback calculations',
)