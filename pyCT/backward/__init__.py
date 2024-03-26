import pyCT
from pyCT.parameter import _Parameters
from .reconstructCPU import *
from reconstructGPU import reconstructParallelBeamGPU, reconstructConeBeamGPU
from copy import deepcopy

def applyFilter(sinogram_array, parameters, filter):
    na = len(parameters.detector.motion.rotation)
    nu = parameters.detector.size.u
    du = parameters.detector.spacing.u
    s2o = parameters.distance.source2object
    s2d = parameters.distance.source2detector

    extended_size = max(64, int(2 ** np.ceil(np.log2(2 * nu))))
    pad = (extended_size - nu) // 2

    n = np.concatenate((np.arange(1, nu + 1, 2, dtype=np.uint32),
                            np.arange(nu - 1, 0, -2, dtype=np.uint32)))
    f = np.zeros(extended_size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2
    fourier_filter = 2 * np.real(np.fft.fft(f))
    fourier_proj = np.fft.fft(np.pad(sinogram_array, [(0,0),(0,0),(pad,pad)]), axis=-1)

    if filter.lower() == 'ramp' or filter.lower()=='ram-lak':
        pass
    elif filter.lower() == 'shepp-logan':
        omega = np.pi * np.fft.fftfreq(extended_size)[1:]
        fourier_filter[1:] *= np.sin(omega) / omega
    elif filter.lower() == 'cosine':
        freq = np.linspace(0, np.pi, extended_size, endpoint=False)
        cosine_filter = np.fft.fftshift(np.sin(freq))
        fourier_filter *= cosine_filter
    elif filter.lower() == 'hamming':
        fourier_filter *= np.fft.fftshift(np.hamming(extended_size))
    elif filter.lower() == 'hann':
        fourier_filter *= np.fft.fftshift(np.hanning(extended_size))

    sinogram_array = np.fft.ifft(fourier_proj*fourier_filter, axis=-1).real
    return sinogram_array[..., pad : -pad] * (np.pi/na/2/du/s2o*s2d)

def reconstruct(sinogram_array : np.ndarray,
                parameters : _Parameters,
                filter = 'ramp',
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
    
    ray_step = 0.5

    # get parameters
    mode = parameters.mode

    near = parameters.distance.near
    far = parameters.distance.far
    s2d = parameters.distance.source2detector
    
    nx, ny, nz = parameters.object.size.get()
    nu, nv = parameters.detector.size.get()
    nw = int((far - near) / ray_step)

    su, sv = parameters.detector.length.get()
    du, dv = parameters.detector.spacing.get()

    na = len(parameters.detector.motion.rotation)
    
    # get transformation
    transformation = pyCT.getTransformation(parameters, nw)
    transformationMatrix = transformation.getBackward()
    
    if filter is None or filter.lower() == 'none':
        pass
    else:
        sinogram_array = applyFilter(sinogram_array, parameters, filter)

    if is_cuda:
        reconstruction_array = np.zeros(nz*ny*nx, dtype=np.float32)
        transformationMatrix = transformationMatrix.flatten().astype(np.float32)
        sinogram_array = sinogram_array.flatten().astype(np.float32)
        if mode:
            reconstruction_array = deepcopy(reconstructConeBeamGPU(reconstruction_array, transformationMatrix, sinogram_array, nx, ny, nz, nu, nv, na, su, sv, du, dv, s2d))
        else:
            reconstruction_array = deepcopy(reconstructParallelBeamGPU(reconstruction_array, transformationMatrix, sinogram_array, nx, ny, nz, nu, nv, na))
        reconstruction_array = reconstruction_array.reshape(nz, ny, nx)
    else:
        reconstruction_array = np.zeros([nz, ny, nx])
        if mode:
            reconstructConeBeamCPU(reconstruction_array, transformationMatrix, sinogram_array, nx, ny, nz, nu, nv, na, su, sv, du, dv, s2d)
        else:
            reconstructParallelBeamCPU(reconstruction_array, transformationMatrix, sinogram_array, nx, ny, nz, nu, nv, na)
    
    return reconstruction_array