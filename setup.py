from setuptools import setup, find_packages

setup(
    name="uwb-sim",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "pylint",
        "pytest",
        "pyqt5",
        "pyqtgraph",
        "pyyaml"
    ],
)
