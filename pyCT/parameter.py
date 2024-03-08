import yaml
import numpy as np
from copy import deepcopy

def getParameters(path_header=None):
    return _Parameters(path_header)

class _Parameters():
    def __init__(self, path_header=None):
        self.mode = None
        self.object = _Object()
        self.detector = _Detector()
        self.distance = _Distance()

        if path_header is not None:
            self.__setParameters(path_header)
        
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
        output +=     'source to object   (mm) : {}\n'.format(self.distance.source2object)
        output +=     'source to detector (mm) : {}\n'.format(self.distance.source2detector)
        return output
    
    def set(self):
        if self.object.size.check() and self.object.spacing.check():
            self.object.length.set([size * spacing 
                                    for size, spacing in zip(self.object.size.get(), 
                                                             self.object.spacing.get())])
        if self.detector.size.check() and self.detector.spacing.check():
            self.detector.length.set([size * spacing 
                                      for size, spacing in zip(self.detector.size.get(), 
                                                             self.detector.spacing.get())])
        if (self.distance.source2object is not None) and (self.distance.source2detector is not None):
            length = .5 * ((self.object.length.x**2 + self.object.length.y**2 + self.object.length.z**2) ** .5)
            self.distance.near = max(0, self.distance.source2object - length)
            self.distance.far = min(self.distance.source2detector, self.distance.source2object + length)
    
    def copy(self):
        return deepcopy(self)
    
    @staticmethod
    def __loadConfig(path):
        with open(path, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)
        
    def __setParameters(self, config_path):
        config = self.__loadConfig(config_path)
        if 'object_size_x' in config.keys() and 'object_spacing_x' in config.keys():
            self.object.length.x = config['object_size_x'] * config['object_spacing_x']
        if 'object_size_y' in config.keys() and 'object_spacing_y' in config.keys():
            self.object.length.y = config['object_size_y'] * config['object_spacing_y']
        if 'object_size_z' in config.keys() and 'object_spacing_z' in config.keys():
            self.object.length.z = config['object_size_z'] * config['object_spacing_z']

        if 'detector_size_u' in config.keys() and 'detector_spacing_u' in config.keys():
            self.detector.length.u = config['detector_size_u'] * config['detector_spacing_u']
        if 'detector_size_v' in config.keys() and 'detector_spacing_v' in config.keys():
            self.detector.length.v = config['detector_size_v'] * config['detector_spacing_v']
        
        if 'distance_source2object' in config.keys() and 'distance_source2detector' in config.keys():
            length = .5 * ((self.object.length.x**2 + self.object.length.y**2 + self.object.length.z**2) ** .5)
            self.distance.near = max(0, config['distance_source2object'] - length)
            self.distance.far = min(config['distance_source2detector'], config['distance_source2object'] + length)

        for key, value in config.items():
            # set mode
            # False: Parallel/Orthographic, True: Cone/Perspective
            if key == 'mode':
                if type(value) is bool:
                    self.mode = value
                
                elif type(value) is int:
                    if value == 0:
                        self.mode = False
                    elif value == 1:
                        self.mode = True

                elif type(value) is str:
                    if value.lower() == 'parallel':
                        self.mode = False
                    elif value.lower() == 'orthographic':
                        self.mode = False
                    elif value.lower() == 'cone':
                        self.mode = True
                    elif value.lower() == 'perspective':
                        self.mode = True
                else:
                    self.mode = None
            
            # set object size
            elif key == 'object_size_x':
                self.object.size.x = value
            elif key == 'object_size_y':
                self.object.size.y = value
            elif key == 'object_size_z':
                self.object.size.z = value
            
            # set object spacing
            elif key == 'object_spacing_x':
                self.object.spacing.x = value
            elif key == 'object_spacing_y':
                self.object.spacing.y = value
            elif key == 'object_spacing_z':
                self.object.spacing.z = value
            
            # set detector size
            elif key == 'detector_size_u':
                self.detector.size.u = value
            elif key == 'detector_size_v':
                self.detector.size.v = value
        
            # set detector spacing
            elif key == 'detector_spacing_u':
                self.detector.spacing.u = value
            elif key == 'detector_spacing_v':
                self.detector.spacing.v = value

            # set distance
            elif key == 'distance_source2object':
                self.distance.source2object = value
            elif key == 'distance_source2detector':
                self.distance.source2detector = value

class _2D():
    def __init__(self, u=None, v=None):
        self.u = u
        self.v = v
    def set(self, ls):
        self.u = ls[0]
        self.v = ls[1]
    def get(self):
        return np.array([self.u, self.v])
    def check(self):
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
    def check(self):
        if (self.x == None) or (self.y == None) or (self.z == None):
            return False
        else:
            return True

class _Motion():
    def __init__(self):
        self.rotation = [[0,0,0],]
        self.translation = [[0,0,0],]
    def set(self, alpha, beta=None, gamma=None, offset=None):
        na = len(alpha)
        if beta is None:
            beta = np.zeros(na)
        if gamma is None:
            gamma = np.zeros(na)
        self.rotation = np.array([alpha, beta, gamma]).T
        if offset is not None:
            self.translation = offset

class _Object():
    def __init__(self):
        self.size = _3D()
        self.length = _3D()
        self.spacing = _3D()
        self.motion = _Motion()

class _Detector():
    def __init__(self):
        self.size = _2D()
        self.length = _2D()
        self.spacing = _2D()
        self.motion = _Motion()
        
class _Distance():
    def __init__(self):
        self.source2object = None
        self.source2detector = None
        self.near = None
        self.far = None
    def check(self):
        if (self.source2object == None) or (self.source2detector == None):
            return False
        else:
            True