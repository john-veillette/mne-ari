from setuptools import setup 

setup(
	name = 'mne_ari',
	version = '0.1.0',
	description = 'All-Resolutions Inference for ME/EEG',
	url = 'https://github.com/john-veillette/mne_ari',
	author = 'John Veillette',
	author_email = 'johnv@uchicago.edu',
	license = 'BSD-3-Clause',
	packages = ['mne_ari'],
	install_requires = ['mne'],
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