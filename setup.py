from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy as np

setup(name='SHCL',
      version='0.1',
      description='Smooth Heteroscedastic Concomitant Lasso',
      author='Under reviewing process :)',
      author_email='xxx@ccc.com',
      url='',
      packages=['shcl'],
      cmdclass={'build_ext': build_ext},
      ext_modules=[
          Extension('shcl.SHCL_fast',
                    sources=['shcl/SHCL_fast.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()]),
      ],
      )
