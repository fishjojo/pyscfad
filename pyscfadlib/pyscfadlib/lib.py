import os
import numpy

def load_library(libname):
    try:
        _loaderpath = os.path.dirname(__file__)
        return numpy.ctypeslib.load_library(libname, _loaderpath)
    except OSError:
        raise

libao2mo_vjp = load_library('libao2mo_vjp')
libcc_vjp = load_library('libcc_vjp')
libcgto_vjp = load_library('libcgto_vjp')
libcvhf_vjp = load_library('libcvhf_vjp')
libnp_helper_vjp = load_library('libnp_helper_vjp')
