import numpy as np
cimport numpy as cnp
cnp.import_array()

# from libc.stdlib cimport malloc, free

# cdef extern from "numpy/arrayobject.h":
#     void PyArray_ENABLEFLAGS(cnp.ndarray arr, int flags)


ctypedef cnp.float32_t DTYPE

cdef extern from "forward.h":
    void funcParallelBeam(float *detector_array, float *transformation, float *object_array, int nx, int ny, int nz, int nu, int nv, int nw, int na)
    void funcConeBeam    (float *detector_array, float *transformation, float *object_array, int nx, int ny, int nz, int nu, int nv, int nw, int na, float su, float sv, float s2d, float near, float far);


def projectParallelBeamGPU(cnp.ndarray[DTYPE, ndim=1] detector_array, cnp.ndarray[DTYPE, ndim=1] transformation, cnp.ndarray[DTYPE, ndim=1] object_array, int nx, int ny, int nz, int nu, int nv, int nw, int na):

    cdef float *c_detector_array = <float *> detector_array.data
    cdef float *c_transformation = <float *> transformation.data
    cdef float *c_object_array = <float *> object_array.data

    funcParallelBeam(c_detector_array, c_transformation, c_object_array, nx, ny, nz, nu, nv, nw, na)

    cdef cnp.npy_intp shape[1]
    shape[0] = <cnp.npy_intp> (na*nv*nu)

    new = cnp.PyArray_SimpleNewFromData(1, shape, cnp.NPY_FLOAT32, c_detector_array)
    #PyArray_ENABLEFLAGS(new, cnp.NPY_OWNDATA)

    return new


def projectConeBeamGPU(cnp.ndarray[DTYPE, ndim=1] detector_array, cnp.ndarray[DTYPE, ndim=1] transformation, cnp.ndarray[DTYPE, ndim=1] object_array, int nx, int ny, int nz, int nu, int nv, int nw, int na, float su, float sv, float s2d, float near, float far):

    cdef float *c_detector_array = <float *> detector_array.data
    cdef float *c_transformation = <float *> transformation.data
    cdef float *c_object_array = <float *> object_array.data

    funcConeBeam(c_detector_array, c_transformation, c_object_array, nx, ny, nz, nu, nv, nw, na, su, sv, s2d, near, far)

    cdef cnp.npy_intp shape[1]
    shape[0] = <cnp.npy_intp> (na*nv*nu)

    new = cnp.PyArray_SimpleNewFromData(1, shape, cnp.NPY_FLOAT32, c_detector_array)
    #PyArray_ENABLEFLAGS(new, cnp.NPY_OWNDATA)

    return new