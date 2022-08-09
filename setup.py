from setuptools import setup, find_packages
import sys 

setup(
    name = 'mne-ari',
    version = '0.1.2',
    description = 'All-Resolutions Inference for M/EEG',
    url = 'https://github.com/john-veillette/mne-ari',
    author = 'John Veillette',
    author_email = 'johnv@uchicago.edu',
    license = 'BSD-3-Clause',
    packages = find_packages(),
    install_requires = ['mne>=0.20', 'numpy>=1.15.4', 'scipy>=1.1.0'],
    classifiers = [
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'License :: OSI Approved',
            'Programming Language :: Python',
            'Topic :: Software Development',
            'Topic :: Scientific/Engineering',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Operating System :: MacOS',
            'Programming Language :: Python :: 3',
    ]
)
