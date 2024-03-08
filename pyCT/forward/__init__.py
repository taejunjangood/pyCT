import pyCT
from pyCT.parameter import _Parameters
from .transformation import *
from .projectCPU import *
from projectGPU import projectParallelBeamGPU, projectConeBeamGPU
from copy import deepcopy

def project(object_array : np.ndarray,
            parameters : _Parameters,
            **kwargs):
    # check CUDA
    is_cuda = False if pyCT.CUDA is None else True        
    if 'cuda' in kwargs.keys():
        if is_cuda:
            if not kwargs['cuda']:
                is_cuda = False
        else:
            if kwargs['cuda']:
                print('CUDA is not available ...')
    
    # set step size
    if 'step' in kwargs.keys():
        step = kwargs['step']
    else:
        step = .5

    # get parameters
    mode = parameters.mode

    near = parameters.distance.near
    far = parameters.distance.far
    s2d = parameters.distance.source2detector
    
    nx, ny, nz = parameters.object.size.get()
    nu, nv = parameters.detector.size.get()
    nw = int((far - near) / step)

    su = parameters.detector.length.u
    sv = parameters.detector.length.v

    na = len(parameters.detector.motion.rotation)
    
    # get transformation
    transformation = getTransformation(parameters)
    transformationMatrix = transformation.get()

    # run
    if is_cuda:
        detector_array = np.zeros(na*nu*nv, dtype=np.float32)
        transformationMatrix = transformationMatrix.flatten().astype(np.float32)
        object_array = object_array.flatten().astype(np.float32)
        if mode:
            detector_array = deepcopy(projectConeBeamGPU(detector_array, transformationMatrix, object_array,  nx, ny, nz, nu, nv, nw, na, su, sv, s2d, near, far))
        else:
            detector_array = deepcopy(projectParallelBeamGPU(detector_array, transformationMatrix, object_array,  nx, ny, nz, nu, nv, nw, na))
        detector_array = detector_array.reshape(na,nv,nu)
    
    else:
        detector_array = np.zeros([na, nv, nu])
        if mode:
            projectConeBeamCPU(detector_array, transformationMatrix, object_array, nx, ny, nz, nu, nv, nw, na, su, sv, s2d, near, far)
        else:
            projectParallelBeamCPU(detector_array, transformationMatrix, object_array, nx, ny, nz, nu, nv, nw, na)
    
    return detector_array * step