import numpy as np
from pyCT.parameter import _Parameters

def getTransformation(params, nw):
    return _Transformation(params, nw)

class _Transformation():
    def __init__(self, params:_Parameters, nw:int):
        self.worldTransformation = None
        self.cameraTransformation = None
        self.viewTransformation = None
        self.__params = params

        self.__setWorldTransformation()
        self.__setCameraTransformation()
        if not params.mode:
            self.__setViewTransformation(nw)

    def getForward(self) -> np.ndarray:
        return np.linalg.inv(self.getBackward())
    
    def getBackward(self) -> np.ndarray:
        if self.__params.mode:
            return np.einsum('aij,ajk->aik', self.cameraTransformation, self.worldTransformation)
        else:
            return self.viewTransformation @ np.einsum('aij,ajk->aik', self.cameraTransformation, self.worldTransformation)


    def __setWorldTransformation(self):
        dx, dy, dz = self.__params.object.spacing.get()
        sx, sy, sz = self.__params.object.length.get()
        self.worldTransformation = np.array([[dx, 0 , 0 , -sx/2+dx/2], 
                                             [0 , dy, 0 , -sy/2+dy/2], 
                                             [0 , 0 , dz, -sz/2+dz/2], 
                                             [0 , 0 , 0 , 1]])
        # [A,4,4]x[4,4] -> [A,4,4]
        self.worldTransformation = np.einsum('aij,jk->aik', _getRotation(self.__params.object.motion.rotation.angles, self.__params.object.motion.rotation.axes), self.worldTransformation)
        self.worldTransformation = np.einsum('aij,ajk->aik', _getTranslation(self.__params.object.motion.translation.vectors), self.worldTransformation)

    
    def __setCameraTransformation(self):
        detectorFrame = np.array([[0,0,1,0],
                                  [1,0,0,0],
                                  [0,1,0,0],
                                  [0,0,0,1]])
        # [A,4,4]x[4,4] -> [A,4,4]
        detectorFrame = np.einsum('aij,jk->aki', _getRotation(self.__params.source.motion.rotation.angles, self.__params.source.motion.rotation.axes), detectorFrame)
        sourceOrigin = self.__params.source.distance.source2object * detectorFrame[:, 2, :3] + self.__params.source.motion.translation.vectors # [A,3]
        self.cameraTransformation = np.einsum('aij,ajk->aik', detectorFrame, _getTranslation(-sourceOrigin))


    def __setViewTransformation(self, nw):
        if self.__params.mode:
            pass
        else:
            nu, nv = self.__params.detector.size.get()
            du, dv = self.__params.detector.spacing.get()
            lengthNear = self.__params.source.distance.near
            lengthFar = self.__params.source.distance.far
            length = (lengthFar - lengthNear)
            viewMatrix = np.array(
                [
                    [1/du, 0   , 0         , -1/2+nu/2],
                    [0   , 1/dv, 0         , -1/2+nv/2],
                    [0   , 0   , -nw/length, -nw*lengthNear/length],
                    [0   , 0   , 0         , 1]
                ]
            )
            motionMatrix = _getRotation(self.__params.detector.motion.rotation.angles, 'z') @ _getTranslation(self.__params.detector.motion.translation.vectors / (-1*self.__params.detector.spacing.get()))
            self.viewTransformation = np.einsum('ij,ajk -> aik', viewMatrix, motionMatrix)


def _makeRotation(angle, axis):
    if type(angle) in [int, float, 
                       np.uint64, np.uint32, np.uint16, np.uint8, np.int64, np. int32, np.int16, np.int8,
                       np.float128, np.float64, np.float32, np.float16]:
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

def _getRotation(angles:np.ndarray, axes:str):
    # angles: [na, nc], axes: [nc] >> out: [na,4,4]
    na = len(angles)
    R = np.eye(4)[None].repeat(na, axis=0)
    # (nc) loops
    for angle, axis in zip(angles.T, axes):
        R = _makeRotation(angle, axis) @ R
    return R
    
def _getTranslation(offset:np.ndarray):
    _, n = offset.shape
    if n == 3:
        R = np.eye(4)[None,...].repeat(n, axis=0)
        R[:, :-1, -1] = offset
    elif n == 2:
        R = np.eye(4)[None,...].repeat(n, axis=0)
        R[:, :-2, -1] = offset
    return R

# def _getTranslation2D(offset:np.ndarray):
#     if offset.shape == (2,):
#         ox, oy = offset
#         return np.array([[1, 0, 0, ox],
#                          [0, 1, 0, oy],
#                          [0, 0, 1, 0],
#                          [0, 0, 0, 1]])
#     elif len(offset.shape) == 2:
#         n, _ = offset.shape
#         R = np.eye(4)[None,...].repeat(n, axis=0)
#         R[:,:-2, -1] = offset
#         return R