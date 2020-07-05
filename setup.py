from setuptools import setup, find_packages

# Get requirements from text file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="ukat",
    version="0.1.0",
    description="UKRIN Kidney Analysis Toolbox",
    url="https://github.com/UKRIN-MAPS/ukat",
    license="GPL-3.0",

    packages=find_packages(),
    # packages=find_packages(exclude=("tests*")), # Want to exclude tests?

    install_requires=requirements,

    package_data={
        # If any package contains files with extensions below, include them:
        "": ["*.json",
             "*.nii.gz",
             "*.bval",
             "*.bval_before_manual_correction",
             "*.bvec",
             "*.png"],
    },
)