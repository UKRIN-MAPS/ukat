from setuptools import setup, find_packages

# Get requirements from text file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Use README.md as the long description
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="ukat",
    version="0.2.0",
    description="UKRIN Kidney Analysis Toolbox",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/UKRIN-MAPS/ukat",
    license="GPL-3.0",

    python_requires='>=3.6, <4',
    packages=find_packages(),
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

    # Classifiers - the purpose is to create a wheel and upload it to PYPI
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Environment :: Console',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate you support Python 3. These classifiers are *not*
        # checked by 'pip install'. See instead 'python_requires' below.
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',

        # Pick your license as you wish
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    ],
)
