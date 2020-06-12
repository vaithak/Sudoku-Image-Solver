# File : setup.py

#from distutils.core import setup, Extension
#name of module
#name  = "SudokuSolve"

#version of module
#version = "1.0"

# specify the name of the extension and source files
# required to compile this
#ext_modules = Extension(name='_SudokuSolve',sources=["sudokuSolve.i","sudokuGen.cpp"]) 

from distutils.core import *
from setuptools import setup, Extension
os.environ["CC"] = "g++" # force compiling c as c++
setup(name='SudokuSolve',
    version='1',
    ext_modules=[Extension('_SudokuSolve', sources=['sudokuSolve.i'],
                    swig_opts=['-c++'],
                    extra_compile_args=['--std=c++14']
                    )],
)
