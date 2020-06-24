from distutils.core import setup

setup(
    name='ukat',
    version='0.1.0',
    description='UKRIN Kidney Analysis Toolbox',
    url='https://github.com/UKRIN-MAPS/ukat',
    license='GPL-3.0',
    packages=['ukat', ],
    install_requires=[
        'numpy>=1.18.1',
        'nibabel>=3.0.2',
        'matplotlib>=3.1.3',
        'scikit-image>=0.16.2',
        'scipy>=1.4.1',
        'dipy>=1.0.0',
    ],
)
