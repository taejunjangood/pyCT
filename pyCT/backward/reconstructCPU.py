import numpy as np


def reconstructParallelBeamCPU(reconstruction_array, transformation, sinogram_array, nx, ny, nz, nu, nv, nw, na):
    pad = 1
    sinogram_array = np.pad(sinogram_array, pad)
    nu += (2*pad -1)
    nv += (2*pad -1)
    for a in range(na):
        matrix = transformation[a]
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    u, v, _, _ = matrix @ [x,y,z,1]
                    u += pad
                    v += pad
                    if 0 < u < nu and 0 < v < nv:
                        u0, v0 = int(u), int(v)
                        u1, v1 = u0 + 1, v0 + 1
                        interp = (v1-v) * ((u1-u)*sinogram_array[v0,u0]+(u-u0)*sinogram_array[v0,u1]) + (v-v0) * ((u1-u)*sinogram_array[v1,u0]+(u-u0)*sinogram_array[v1,u1])
                        reconstruction_array[a][z][y][x] += interp