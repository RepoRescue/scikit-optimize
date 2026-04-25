from setuptools import setup
import builtins

# This is a bit (!) hackish: we are setting a global variable so that the
# main skopt __init__ can detect if it is being loaded by the setup
# routine
builtins.__SKOPT_SETUP__ = True

import skopt

VERSION = skopt.__version__

CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved :: BSD License',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.13']


setup(name='scikit-optimize',
      version=VERSION,
      description='Sequential model-based optimization toolbox.',
      long_description=open('README.rst').read(),
      url='https://scikit-optimize.github.io/',
      license='BSD 3-clause',
      author='The scikit-optimize contributors',
      classifiers=CLASSIFIERS,
      packages=['skopt', 'skopt.learning', 'skopt.optimizer', 'skopt.space',
                'skopt.learning.gaussian_process', 'skopt.sampler'],
      install_requires=['joblib>=1.5.3', 'pyaml>=26.2.1', 'numpy>=2.4.3',
                        'scipy>=1.17.1',
                        'scikit-learn>=1.8.0'],
      extras_require={
        'plots':  ["matplotlib>=3.10.8"]
        },
      python_requires='>=3.13'

      )
