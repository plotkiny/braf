from setuptools import setup

setup(
    name='BRAF',
    version='0.1',
    packages=['braf',],
    long_description=open('README.txt').read(),
    install_requires=open('requirements.txt').read().split(),
)
