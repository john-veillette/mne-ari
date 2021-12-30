from setuptools import setup 
from os.path import realpath, dirname, join
import sys 

PROJECT_ROOT = dirname(realpath(__file__))
REQUIREMENTS_FILE = join(PROJECT_ROOT, 'requirements.txt')

with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()

setup(
    name = 'mne_ari',
    version = '0.1.0',
    description = 'All-Resolutions Inference for ME/EEG',
    url = 'https://github.com/john-veillette/mne_ari',
    author = 'John Veillette',
    author_email = 'johnv@uchicago.edu',
    license = 'BSD-3-Clause',
    packages = ['mne_ari'],
    install_requires = install_reqs,
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