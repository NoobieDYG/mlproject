from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    with open(file_path, 'r') as file:
        lines = file.readlines()
    requirements = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            requirements.append(line)
    return requirements

setup(
    name='mlprojcet',
    version='0.0.1',
    author='me',
    author_email='me@me',
    packages=find_packages(),
    install_requires=get_requirements('poetry.lock'),

)