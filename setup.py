from distutils.core import setup
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

import os
thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        # install_requires = f.read().splitlines()
        install_requires = [x.replace("'","") for x in f.read().split(', ')]

# print(install_requires)

setup(
    name="RT-203-Components",
    version="3.3",
    packages=["components"],
    license="DO NOT COPY!",
    description="Components for RT 203 SERC Project",
    install_requires=install_requires,
)
