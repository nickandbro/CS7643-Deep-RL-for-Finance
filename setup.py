import sys
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

src_folder_path = "src"
test_folder_path = "tests"

with open("README.md", "r") as fh:
    long_desc = fh.read()

try:
    REQUIRES = []
    f = open("requirements.txt", "rb")
    for line in f.read().decode("utf-8").split("\n"):
        line = line.strip()
        if "#" in line:
            line = line[: line.find("#")].strip()
        if line:
            REQUIRES.append(line)
except FileNotFoundError:
    print("'requirements.txt' not found!")
    REQUIRES = list()

setup(
    name="CS7643-Deep-RL-for-Finance",
    version="0.0.1",
    include_package_data=True,
    author="Taylor Last, Nick Brooks, Pol John Cruz, Daniel Hall",
    author_email="tlast3@gatech.edu",
    url="https://github.com/nickandbro/CS7643-Deep-RL-for-Finance",
    # license="MIT",
    packages=find_packages(),
    description="Deep Reinforcement Learning for Financial Portfolio Allocation",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="Reinforcement Learning, Finance, Portfolio Allocation, DeepRL"
    platform=["any"],
    python_requires=">=3.7",
)