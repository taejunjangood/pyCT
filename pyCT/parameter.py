import json
import numpy as np
from copy import deepcopy

def getParameters(path_header=None):
    return _Parameters(path_header)

class _Parameters():
    def __init__(self, path_params:str):
        self.mode = None
        self.object = _Object()
        self.detector = _Detector()
        self.source = _Source()

        if path_params is not None:
            self.load(path_params)
        
    def __str__(self):
        output = ''
        if self.mode:
            output += 'mode                    : cone\n'
        else:
            output += 'mode                    : parallel\n'
        output +=     'object size     (voxel) : ({}, {} ,{})\n'.format(self.object.size.x, self.object.size.y, self.object.size.z)
        output +=     'object spacing     (mm) : ({}, {} ,{})\n'.format(self.object.spacing.x, self.object.spacing.y, self.object.spacing.z)
        output +=     'object length      (mm) : ({}, {} ,{})\n'.format(self.object.length.x, self.object.length.y, self.object.length.z)
        output +=     'detector size   (pixel) : ({}, {})\n'.format(self.detector.size.u, self.detector.size.v)
        output +=     'detector spacing   (mm) : ({}, {})\n'.format(self.detector.spacing.u, self.detector.spacing.v)
        output +=     'detector length    (mm) : ({}, {})\n'.format(self.detector.length.u, self.detector.length.v)
        output +=     'source to object   (mm) : {}\n'.format(self.source.distance.source2object)
        output +=     'source to detector (mm) : {}\n'.format(self.source.distance.source2detector)
        return output
    
    def load(self, path:str):
        meta = json.load(open(path, 'r'))
        
        self.mode = meta['mode']
        # object
        self.object.size.set(list(meta['object']['size'].values()))
        self.object.spacing.set(list(meta['object']['spacing'].values()))
        self.object.length.set(list(meta['object']['length'].values()))
        self.object.motion.rotation.angles = np.array(meta['object']['motion']['rotation']['angles'])
        self.object.motion.rotation.axes = np.array(meta['object']['motion']['rotation']['axes'])
        self.object.motion.translation.vectors = np.array(meta['object']['motion']['translation']['vectors'])
        # detector
        self.detector.size.set(list(meta['detector']['size'].values()))
        self.detector.spacing.set(list(meta['detector']['spacing'].values()))
        self.detector.length.set(list(meta['detector']['length'].values()))
        self.detector.motion.rotation.angles = np.array(meta['detector']['motion']['rotation']['angles'])
        self.detector.motion.translation.vectors = np.array(meta['detector']['motion']['translation']['vectors'])
        # source
        self.source.distance.source2object = meta['source']['distance']['source2object']
        self.source.distance.source2detector = meta['source']['distance']['source2detector']
        self.source.distance.near = meta['source']['distance']['near']
        self.source.distance.far = meta['source']['distance']['far']
        self.source.motion.rotation.angles = np.array(meta['source']['motion']['rotation']['angles'])
        self.source.motion.rotation.axes = np.array(meta['source']['motion']['rotation']['axes'])
        self.source.motion.translation.vectors = np.array(meta['source']['motion']['translation']['vectors'])

    def save(self, path:str):
        meta = deepcopy(self.__dict__)
        stack = [meta]
        while stack:
            element = stack.pop()
            for key, value in element.items():
                if type(value) in [bool, int, float, str, list, tuple]:
                    pass
                elif type(value) == np.ndarray:
                    element[key] = value.tolist()
                elif type(value) in [np.float128,np.float64, np.float32, np.float16]:
                    element[key] = float(value)
                elif type(value) in [np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint16, np.uint8]:
                    element[key] = int(value)
                else:
                    element[key] = value.__dict__
                    stack.append(element[key])
        json.dump(meta, open(path, 'w'), indent=2)

    def set(self):
        if self.object.size.checkNone() and self.object.spacing.checkNone():
            self.object.length.set([size * spacing 
                                    for size, spacing in zip(self.object.size.get(), 
                                                             self.object.spacing.get())])
        if self.detector.size.checkNone() and self.detector.spacing.checkNone():
            self.detector.length.set([size * spacing 
                                      for size, spacing in zip(self.detector.size.get(), 
                                                             self.detector.spacing.get())])
        if self.source.distance.checkNone():
            length = .5 * ((self.object.length.x**2 + self.object.length.y**2 + self.object.length.z**2) ** .5)
            self.source.distance.near = max(0, self.source.distance.source2object - length)
            self.source.distance.far = min(self.source.distance.source2detector, self.source.distance.source2object + length)
        
    def copy(self):
        return deepcopy(self)
        

class _2D():
    def __init__(self, u=None, v=None):
        self.u = u
        self.v = v
    def set(self, ls):
        self.u = ls[0]
        self.v = ls[1]
    def get(self):
        return np.array([self.u, self.v])
    def checkNone(self):
        if (self.u == None) or (self.v == None):
            return False
        else:
            return True

class _3D():
    def __init__(self, x=None, y=None, z=None):
        self.x = x
        self.y = y
        self.z = z
    def set(self, ls):
        self.x = ls[0]
        self.y = ls[1]
        self.z = ls[2]
    def get(self):
        return np.array([self.x, self.y, self.z])
    def checkNone(self):
        if (self.x == None) or (self.y == None) or (self.z == None):
            return False
        else:
            return True

# class _Motion2D():
#     def __init__(self):
#         self.rotation = np.array([0,])
#         self.translation = np.array([[0,0],])
#     def set(self, alpha, offset=None):
#         if type(alpha) in [int,float]:
#             alpha = np.array([alpha])
#         elif type(alpha) != np.ndarray:
#             alpha = np.array(alpha)
#         if len(alpha.shape) != 1:
#             raise ValueError('Alpha is not inappropriate.')
#         self.rotation = alpha
        
#         if offset is not None:
#             if type(offset) != np.ndarray:
#                 offset = np.array(offset)
#             if offset.shape == (2,):
#                 offset = offset[None]
#             elif len(offset.shape) != 2 or offset.shape[1] != 2:
#                 raise ValueError('Offset is not inappropriate.')
#             self.translation = offset

# class _Motion():
#     def __init__(self):
#         self.rotation = np.array([[0,0,0],])
#         self.translation = np.array([[0,0,0],])
#     def set(self, alpha, beta=None, gamma=None, offset=None):
#         if type(alpha) in [int, float]:
#             alpha = np.array([alpha])
#         elif type(alpha) != np.ndarray:
#             alpha = np.array(alpha)
#         if len(alpha.shape) != 1:
#             raise ValueError('Alpha is not inappropriate.')
#         na = len(alpha)
#         if beta is None:
#             beta = np.zeros(na)
#         if gamma is None:
#             gamma = np.zeros(na)
#         self.rotation = np.stack([alpha, beta, gamma], axis=-1)

#         if offset is not None:
#             if type(offset) != np.ndarray:
#                 offset = np.array(offset)
#             if offset.shape == (3,):
#                 offset = offset[None]
#             elif len(offset.shape) !=2 or offset.shape[1] != 3:
#                 raise ValueError('Offset is not inappropriate.')
#             self.translation = offset

class _Motion():
    def __init__(self, dim):
        
        if dim==2:
            self.rotation = _Rotation2D()
        elif dim==3:
            self.rotation = _Rotation3D()            
        else:
            raise ValueError()
        self.translation = _Translation(dim)

    def checkSize(self):
        if (len(self.rotation.angles) != len(self.translation.vectors)) and (len(self.translation.vectors) != 1):
            return False
        else:
            return True

class _Rotation2D():
    def __init__(self):
        self.angles = np.zeros([1])
    def set(self, angles:np.ndarray):
        if type(angles) in [list, tuple]:
            angles = np.array(angles)
        if len(angles.shape) != 1:
            raise ValueError()
        self.angles = angles

class _Rotation3D():
    def __init__(self):
        self.angles = np.zeros([1,1])
        self.axes = 'z'
    def set(self, *angles:np.ndarray, axes:str='z'):
        if len(angles)!=len(axes):
            print(len(angles), len(axes))
            raise ValueError()
        if all([len(angle)-len(angles[0]) for angle in angles]):
            raise ValueError()
        for axis in axes:
            if axis.lower() not in ['x', 'y', 'z']:
                raise ValueError()
        self.angles = np.stack(angles, axis=1)
        self.axes = axes

class _Translation():
    def __init__(self, dim):
        self.vectors = np.zeros([1,dim])
    def set(self, vectors:np.ndarray):
        if (type(vectors) in [list, tuple]) and (len(vectors) == 2) and all([type(i) in [int, float] for i in vectors]):
            self.vectors = np.array([vectors])
        elif (type(vectors) == np.ndarray) and (vectors.shape == (2,)):
            self.vectors = vectors[None]
        else:
            raise ValueError()
        

class _Distance():
    def __init__(self):
        self.source2object = None
        self.source2detector = None
        self.near = None
        self.far = None
    def checkNone(self):
        if (self.source2object == None) or (self.source2detector == None):
            return False
        else:
            return True

class _Object():
    def __init__(self):
        self.size = _3D()
        self.length = _3D()
        self.spacing = _3D()
        self.motion = _Motion(dim=3)

class _Detector():
    def __init__(self):
        self.size = _2D()
        self.length = _2D()
        self.spacing = _2D()
        self.motion = _Motion(dim=2)
        
class _Source():
    def __init__(self):
        self.distance = _Distance()
        self.motion = _Motion(dim=3)