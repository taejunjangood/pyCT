import pyCT
from pyCT.parameter import _Parameters
from .reconstructCPU import *
# from reconstructGPU import reconstrucParallelBeamGPU, reconstrucConeBeamGPU
from copy import deepcopy

def reconstruct(sinogram_array : np.ndarray,
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
