import numpy as np
from pyCT.parameter import _Parameters

def getTransformation(params):
    return _Transformation(params)

class _Transformation():
    def __init__(self, params : _Parameters):
        self.worldTransformation = None
        self.cameraTransformation = None
        self.viewTransformation = None
        self.__params = params

        self.__setWorldTransformation()
        self.__setCameraTransformation()
        if not params.mode:
            self.__setViewTransformation()


    def get(self):
        if self.__params.mode:
            return np.linalg.inv(np.einsum('aij,ajk->aik', self.cameraTransformation, self.worldTransformation))
        else:
            return np.linalg.inv(self.viewTransformation @ np.einsum('aij,ajk->aik', self.cameraTransformation, self.worldTransformation))


    def __setWorldTransformation(self):
        dx, dy, dz = self.__params.object.spacing.get()
        sx, sy, sz = self.__params.object.length.get()
        self.worldTransformation = np.array([[dx, 0 , 0 , -sx/2+dx/2], 
                                             [0 , dy, 0 , -sy/2+dy/2], 
                                             [0 , 0 , dz, -sz/2+dz/2], 
                                             [0 , 0 , 0 , 1]])
        # [A,4,4]x[4,4] -> [A,4,4]
        self.worldTransformation = np.einsum('aij,jk->aik', _getRotation(self.__params.object.motion.rotation, 'zxz'), self.worldTransformation)
        self.worldTransformation = np.einsum('aij,ajk->aik', _getTranslation(self.__params.object.motion.translation), self.worldTransformation)

    
    def __setCameraTransformation(self):
        detectorFrame = np.array([[0,0,1,0],
                                  [1,0,0,0],
                                  [0,1,0,0],
                                  [0,0,0,1]])
        # [A,4,4]x[4,4] -> [A,4,4]
        detectorFrame = np.einsum('aij,jk->aki', _getRotation(self.__params.detector.motion.rotation, 'zxz'), detectorFrame)
        sourceOrigin = self.__params.distance.source2object * detectorFrame[:, 2, :3] + self.__params.detector.motion.translation # [A,3]
        self.cameraTransformation = np.einsum('aij,ajk->aik', detectorFrame, _getTranslation(-sourceOrigin))
        
    
    def __setViewTransformation(self):
        if self.__params.mode:
            pass
        else:
            halfWidth = self.__params.detector.length.u / 2
            halfHeight = self.__params.detector.length.v / 2
            lengthNear = self.__params.distance.near
            lengthFar = self.__params.distance.far
            self.viewTransformation = np.array([[1/halfWidth, 0, 0, 0], 
                                        [0, 1/halfHeight, 0, 0], 
                                        [0, 0, -2/(lengthFar-lengthNear), -(lengthFar+lengthNear)/(lengthFar-lengthNear)],
                                        [0, 0, 0, 1]])


def _makeRotation(angle, axis):
    if type(angle) in [int, float]:
        if axis == 'z':
            return np.array([[np.cos(angle), -np.sin(angle), 0, 0],
                             [np.sin(angle),  np.cos(angle), 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
        elif axis == 'y':
            return np.array([[np.cos(angle), 0, -np.sin(angle), 0],
                             [0, 1, 0, 0],
                             [np.sin(angle), 0, np.cos(angle), 0],
                             [0, 0, 0, 1]])
        elif axis == 'x':
            return np.array([[1, 0, 0, 0],
                             [0, np.cos(angle), -np.sin(angle), 0],
                             [0, np.sin(angle), np.cos(angle), 0],
                             [0, 0, 0, 1]])
        
    elif type(angle) in [list, tuple, np.ndarray]:
        na = len(angle)
        if axis == 'z':
            return np.array([[np.cos(angle), -np.sin(angle), np.zeros(na), np.zeros(na)],
                             [np.sin(angle),  np.cos(angle), np.zeros(na), np.zeros(na)],
                             [np.zeros(na), np.zeros(na), np.ones(na), np.zeros(na)],
                             [np.zeros(na), np.zeros(na), np.zeros(na), np.ones(na)]]).transpose(2,0,1)
        elif axis == 'y':
            return np.array([[np.cos(angle), np.zeros(na), -np.sin(angle), np.zeros(na)],
                             [np.zeros(na), np.ones(na), np.zeros(na), np.zeros(na)],
                             [np.sin(angle), np.zeros(na), np.cos(angle), np.zeros(na)],
                             [np.zeros(na), np.zeros(na), np.zeros(na), np.ones(na)]]).transpose(2,0,1)
        elif axis == 'x':
            return np.array([[np.ones(na), np.zeros(na), np.zeros(na), np.zeros(na)],
                             [np.zeros(na), np.cos(angle), -np.sin(angle), np.zeros(na)],
                             [np.zeros(na), np.sin(angle), np.cos(angle), np.zeros(na)],
                             [np.zeros(na), np.zeros(na), np.zeros(na), np.ones(na)]]).transpose(2,0,1)
        else:
            raise ValueError('axis must be entered by one in {x, y, z}.')

def _getRotation(angles, axes):
    # angles: [na, nc], axes: [na] >> out: [na,4,4]
    if type(angles) is not np.ndarray:
        angles = np.array(angles)
    na, _ = angles.shape
    R = np.eye(4)[None].repeat(na, axis=0)
    # (nc) loops
    for angle, axis in zip(angles.T, axes):
        R = _makeRotation(angle, axis) @ R
    return R
    
def _getTranslation(offset):
    if np.array(offset).shape == (3,):
        ox, oy, oz = offset
        return np.array([[1, 0, 0, ox],
                         [0, 1, 0, oy],
                         [0, 0, 1, oz],
                         [0, 0, 0, 1]])
    elif len(np.array(offset).shape) == 2:
        n, _ = np.array(offset).shape
        R = np.eye(4)[None,...].repeat(n, axis=0)
        R[:,:-1, -1] = offset
        return R