# from distutils.core import setup
# from Cython.Build import cythonize


# files = "constraint.pyx", "point.pyx", "cloth.pyx", "shapecloth.pyx", "circlecloth.pyx", "mouse.pyx", "tensioner.pyx", "scorer.pyx"

# for file in files:
# 	setup(ext_modules = cythonize(file))

# # run in terminalpython setup.py build_ext --inplace
# # python setup.py build_ext --inplace



from setuptools import setup, Extension
from Cython.Build import cythonize
import os

# فایل‌های .pyx که باید کامپایل شوند
files = [
    "shapecloth.pyx"]


# تعریف اکستنشن‌ها
extensions = [Extension(file.split(".")[0], [file]) for file in files]

# تنظیمات setup
setup(
    name="CythonModules",
    ext_modules=cythonize(extensions),
)

#python setup.py build_ext --inplace

# python setup.py build_ext --build-lib artifacts/to/ --inplace