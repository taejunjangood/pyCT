import numpy as np

def getTransformation(params):
    return _Transformation(params)

class _Transformation():
    def __init__(self, params):
        self.worldTransformation = None
        self.cameraTransformation = None
        self.viewTransformation = None
        
        self.__params = params
        self.__setWorldTransformation()
        if not params.mode:
            self.__setViewTransformation()

    def getMatrix(self, angle):
        self.__setCameraTransformation(angle)
        if self.__params.mode:
            return np.linalg.inv(self.cameraTransformation @ self.worldTransformation)
        else:
            return np.linalg.inv(self.viewTransformation @ self.cameraTransformation @ self.worldTransformation)


    def __setWorldTransformation(self):
        dx = self.__params.object.spacing.x
        dy = self.__params.object.spacing.y
        dz = self.__params.object.spacing.z
        sx = self.__params.object.length.x
        sy = self.__params.object.length.y
        sz = self.__params.object.length.z
        self.worldTransformation = np.array([[dx, 0, 0, -sx/2+dx/2], 
                                        [0, dy, 0, -sy/2+dy/2], 
                                        [0, 0, dz, -sz/2+dz/2], 
                                        [0, 0, 0, 1]])
           
    
    def __setCameraTransformation(self, angles):
        if type(angles) is int or type(angles) is float:
            angles = np.array([angles])
        a = len(angles)
        R1 = np.array([[0,0,1],[1,0,0],[0,1,0]])
        R2 = np.array([[np.cos(angles), -np.sin(angles), np.zeros(a)],
                         [np.sin(angles), np.cos(angles), np.zeros(a)],
                         [np.zeros(a), np.zeros(a), np.ones(a)]]) # [3,3,A]
        detectorFrame = np.einsum('ija,jk->ika', R2, R1).T # [3,3,A]x[3,3] -> [3,3,A] -> [A,3,3]
        sourceOrigin = self.__params.distance.source2object * detectorFrame[:,2] # [A,3]
        
        translation = np.eye(4)[None,...].repeat(a, axis=0)
        translation[:,:-1, -1] = -sourceOrigin
        rotation = np.eye(4)[None,...].repeat(a, axis=0)
        rotation[:, :-1, :-1] = detectorFrame
        self.cameraTransformation = np.einsum('aij,ajk->aik', rotation, translation)

        
    
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