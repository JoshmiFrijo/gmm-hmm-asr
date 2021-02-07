from setuptools import find_packages
from setuptools import setup

from glob import glob
from os.path import splitext, basename


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gmm-hmm-asr", # Replace with your own username
    version="0.0.1",
    author="Desh Raj",
    author_email="r.desh26@gmail.com",
    description="Simple GMM-HMM models for isolated digit recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/desh2608/gmm-hmm-asr",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.20.0',
    ],
    tests_require=[
        'pytest>=6.2.2'
        'pytest-cov>=2.11.1'
    ]
)