from setuptools import setup, find_packages
from pathlib import Path

requirements_file = Path('requirements.txt')

if (not requirements_file.exists()) or (not requirements_file.is_file()):
    raise Exception("No requirements.txt found")
with open(requirements_file) as f:
    install_requires = list(f.read().splitlines())

setup(
    name='boxes',
    version='0.0.1',
    description='PyTorch Boxes',
    packages=find_packages(
        'boxes', exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    package_data={'boxes': ['py.typed']},
    install_requires=install_requires,
    zip_safe=False)
