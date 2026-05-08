from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        "autoencoder.pyx",
        annotate=True,
        compiler_directives={"language_level": "3"},
    )
)