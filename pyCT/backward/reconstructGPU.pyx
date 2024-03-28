import numpy as np
cimport numpy as cnp
cnp.import_array()

ctypedef cnp.float32_t DTYPE

cdef extern from "backward.h":
    void funcParallelBeam(float *reconstruction_array, float *transformation, float *sinogram_array, int nx, int ny, int nz, int nu, int nv, int na)
    void funcConeBeam(float *reconstruction_array, float *transformation, float *sinogram_array, int nx, int ny, int nz, int nu, int nv, int na, float su, float sv, float du, float dv, float ou, float ov, float s2d);

def reconstructParallelBeamGPU(cnp.ndarray[DTYPE, ndim=1] reconstruction_array, cnp.ndarray[DTYPE, ndim=1] transformation, cnp.ndarray[DTYPE, ndim=1] sinogram_array, int nx, int ny, int nz, int nu, int nv, int na):

    cdef float *c_reconstruction_array = <float *> reconstruction_array.data
    cdef float *c_transformation = <float *> transformation.data
    cdef float *c_sinogram_array = <float *> sinogram_array.data

    funcParallelBeam(c_reconstruction_array, c_transformation, c_sinogram_array, nx, ny, nz, nu, nv, na)

    cdef cnp.npy_intp shape[1]
    shape[0] = <cnp.npy_intp> (nx*ny*nz)

    new = cnp.PyArray_SimpleNewFromData(1, shape, cnp.NPY_FLOAT32, c_reconstruction_array)

    return new

def reconstructConeBeamGPU(cnp.ndarray[DTYPE, ndim=1] reconstruction_array, cnp.ndarray[DTYPE, ndim=1] transformation, cnp.ndarray[DTYPE, ndim=1] sinogram_array, int nx, int ny, int nz, int nu, int nv, int na, float su, float sv, float du, float dv, float ou, float ov, float s2d):

    cdef float *c_reconstruction_array = <float *> reconstruction_array.data
    cdef float *c_transformation = <float *> transformation.data
    cdef float *c_sinogram_array = <float *> sinogram_array.data

    funcConeBeam(c_reconstruction_array, c_transformation, c_sinogram_array, nx, ny, nz, nu, nv, na, su, sv, du, dv, ou, ov, s2d)

    cdef cnp.npy_intp shape[1]
    shape[0] = <cnp.npy_intp> (nx*ny*nz)

    new = cnp.PyArray_SimpleNewFromData(1, shape, cnp.NPY_FLOAT32, c_reconstruction_array)

    return new