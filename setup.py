from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy as np

setup(name='SHCL',
      version='0.1',
      description='Smooth Heteroscedastic Concomitant Lasso',
      author='Mathurin Massias',
      author_email='mathurin.massias@gmail.com',
      url='',
      packages=['shcl'],
      cmdclass={'build_ext': build_ext},
      ext_modules=[
          Extension('shcl.singletask_fast',
                    sources=['shcl/singletask_fast.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()]),
          Extension('shcl.multitask_fast',
                    sources=['shcl/multitask_fast.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()]),
      ],
      )
