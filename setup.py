from setuptools import setup, find_packages
from typing import List

HDE = '-e .'

def get_requirements(path:str)->List:
    requirements= []
    with open(path,'r') as file:
        requirements = file.readlines()
    requirements = [req.replace("\n","") for req in requirements]

    if HDE in requirements:
        requirements.remove(HDE)

    return requirements


setup(
    name = "Student performance prediction",
    version= "0.0.1",
    author= 'Vraj Bhavsar',
    author_email='vrajcbhavsar0905@gmail.com',
    packages=find_packages(),
    install_requirements = get_requirements('requirements.txt')
)