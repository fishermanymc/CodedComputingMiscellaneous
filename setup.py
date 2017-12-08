from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os
import sys
import shutil
import subprocess
import numpy


def main():

    libname = 'myXOR'
    # libname = 'nccore_old_api'

    # Clean auxiliary build files.
    os.getcwd()
    if 'clean' in sys.argv:
        tmp = './build/'
        print('Removing build directory <{}>'.format(tmp))
        shutil.rmtree(tmp, ignore_errors=True)

        tmp = '{}.egg-info'.format(libname)
        print('Removing egg directory <{}>'.format(tmp))
        shutil.rmtree(tmp, ignore_errors=True)

        # Remove Cython generated cpp file.
        subprocess.call('rm {}.cpp'.format(libname), shell=True)

        # If the extension module was built with
        #   >> python setup.py build_ext --inplace
        # then there will be a dynamic library next to
        # the pyx file - delete it.
        subprocess.call('rm {}.cpython-*.so'.format(libname), shell=True)

        # Do not proceed to the build part.
        return

    ext_modules = [Extension(libname, [libname + ".pyx", "myXor.cpp"],
                   language='c++', extra_compile_args=["-std=c++11"],
                   include_dirs=[numpy.get_include()])]

    setup(cmdclass={'build_ext': build_ext}, ext_modules=ext_modules)

if __name__ == '__main__':
    main()