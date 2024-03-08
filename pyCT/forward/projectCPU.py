import numpy as np

def projectConeBeamCPU(detector_array, transformation, object_array, nx, ny, nz, nu, nv, nw, na, su, sv, s2d, near, far):
    pad = 1
    object_array = np.pad(object_array, pad)
    nx += (2*pad -1)
    ny += (2*pad -1)
    nz += (2*pad -1)    
    X, Y = np.meshgrid(np.linspace(-su/2+(su/nu)/2, su/2-(su/nu)/2, nu), np.linspace(-sv/2+(sv/nv)/2, sv/2-(sv/nv)/2, nv))
    Z = -s2d * np.ones_like(X)
    directions = np.stack([X,Y,Z], axis=-1)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    for ia in range(na):
        A = transformation[ia]
        for iu in range(nu):
            for iv in range(nv):
                direction = directions[iv,iu]
                for iw in range(nw):
                    x, y, z = (near + iw*(far-near)/nw) * direction
                    x, y, z, _ = A @ [x,y,z,1] + pad
                    if 0 < x < nx and 0 < y < ny and 0 < z < nz:
                        x0, y0, z0 = int(x), int(y), int(z)
                        x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
                        interp = (z1 - z) * ((y1 - y) * ((x1 - x) * object_array[z0, y0, x0] + (x - x0) * object_array[z0, y0, x1]) + (y - y0) * ((x1 - x) * object_array[z0, y1, x0] + (x - x0) * object_array[z0, y1, x1])) + (z - z0) * ((y1 - y) * ((x1 - x) * object_array[z1, y0, x0] + (x - x0) * object_array[z1, y0, x1]) + (y - y0) * ((x1 - x) * object_array[z1, y1, x0] + (x - x0) * object_array[z1, y1, x1]))
                        detector_array[ia][iv][iu] += interp


def projectParallelBeamCPU(detector_array, transformation, object_array, nx, ny, nz, nu, nv, nw, na):
    pad = 1
    object_array = np.pad(object_array, pad)
    nx += (2*pad -1)
    ny += (2*pad -1)
    nz += (2*pad -1)    
    for ia in range(na):
        matrix = transformation[ia]
        for iu in range(nu):
            u = -1 + (1+iu*2)/nu
            for iv in range(nv):
                v = -1 + (1+iv*2)/nv
                for iw in range(nw):
                    w = -1 + (1+iw*2)/nw
                    x, y, z, _ = matrix @ [u,v,w,1] + pad
                    if 0 < x < nx and 0 < y < ny and 0 < z < nz:
                        x0, y0, z0 = int(x), int(y), int(z)
                        x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
                        interp = (z1 - z) * ((y1 - y) * ((x1 - x) * object_array[z0, y0, x0] + (x - x0) * object_array[z0, y0, x1]) + (y - y0) * ((x1 - x) * object_array[z0, y1, x0] + (x - x0) * object_array[z0, y1, x1])) + (z - z0) * ((y1 - y) * ((x1 - x) * object_array[z1, y0, x0] + (x - x0) * object_array[z1, y0, x1]) + (y - y0) * ((x1 - x) * object_array[z1, y1, x0] + (x - x0) * object_array[z1, y1, x1]))
                        detector_array[ia][iv][iu] += interp