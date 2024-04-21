import yaml, json
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
        self.object.motion.rotation = np.array(meta['object']['motion']['rotation'])
        self.object.motion.translation = np.array(meta['object']['motion']['translation'])
        # detector
        self.detector.size.set(list(meta['detector']['size'].values()))
        self.detector.spacing.set(list(meta['detector']['spacing'].values()))
        self.detector.length.set(list(meta['detector']['length'].values()))
        self.detector.motion.rotation = np.array(meta['detector']['motion']['rotation'])
        self.detector.motion.translation = np.array(meta['detector']['motion']['translation'])
        # source
        self.source.distance.source2object = meta['source']['distance']['source2object']
        self.source.distance.source2detector = meta['source']['distance']['source2detector']
        self.source.distance.near = meta['source']['distance']['near']
        self.source.distance.far = meta['source']['distance']['far']
        self.source.motion.rotation = np.array(meta['source']['motion']['rotation'])
        self.source.motion.translation = np.array(meta['source']['motion']['translation'])

    def save(self, path:str):
        meta = self.__dict__
        list_meta = [meta]
        while list_meta:
            temp_meta = list_meta.pop()
            for key, value in temp_meta.items():
                if type(value) in [bool, int, float, str, list, tuple]:
                    pass
                elif type(value) == np.ndarray:
                    temp_meta[key] = value.tolist()
                elif type(value) in [np.float128,np.float64, np.float32, np.float16]:
                    temp_meta[key] = float(value)
                elif type(value) in [np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint16, np.uint8]:
                    temp_meta[key] = int(value)
                else:
                    temp_meta[key] = value.__dict__
                    list_meta.append(temp_meta[key])
        json.dump(meta, open(path, 'w'), indent=2)

    def set(self):
        if self.object.size.check() and self.object.spacing.check():
            self.object.length.set([size * spacing 
                                    for size, spacing in zip(self.object.size.get(), 
                                                             self.object.spacing.get())])
        if self.detector.size.check() and self.detector.spacing.check():
            self.detector.length.set([size * spacing 
                                      for size, spacing in zip(self.detector.size.get(), 
                                                             self.detector.spacing.get())])
        if self.source.distance.check():
            length = .5 * ((self.object.length.x**2 + self.object.length.y**2 + self.object.length.z**2) ** .5)
            self.source.distance.near = max(0, self.source.distance.source2object - length)
            self.source.distance.far = min(self.source.distance.source2detector, self.source.distance.source2object + length)
        
    def copy(self):
        return deepcopy(self)
        
    # def __setParameters(self, config_path):
    #     with open(config_path, 'r') as f:
    #         config = yaml.load(f, Loader=yaml.FullLoader)

    #     if 'object_size_x' in config.keys() and 'object_spacing_x' in config.keys():
    #         self.object.length.x = config['object_size_x'] * config['object_spacing_x']
    #     if 'object_size_y' in config.keys() and 'object_spacing_y' in config.keys():
    #         self.object.length.y = config['object_size_y'] * config['object_spacing_y']
    #     if 'object_size_z' in config.keys() and 'object_spacing_z' in config.keys():
    #         self.object.length.z = config['object_size_z'] * config['object_spacing_z']

    #     if 'detector_size_u' in config.keys() and 'detector_spacing_u' in config.keys():
    #         self.detector.length.u = config['detector_size_u'] * config['detector_spacing_u']
    #     if 'detector_size_v' in config.keys() and 'detector_spacing_v' in config.keys():
    #         self.detector.length.v = config['detector_size_v'] * config['detector_spacing_v']
        
    #     if 'distance_source2object' in config.keys() and 'distance_source2detector' in config.keys():
    #         length = .5 * ((self.object.length.x**2 + self.object.length.y**2 + self.object.length.z**2) ** .5)
    #         self.source.distance.near = max(0, config['distance_source2object'] - length)
    #         self.source.distance.far = min(config['distance_source2detector'], config['distance_source2object'] + length)

    #     for key, value in config.items():
    #         # set mode
    #         # False: Parallel/Orthographic, True: Cone/Perspective
    #         if key == 'mode':
    #             if type(value) is bool:
    #                 self.mode = value
                
    #             elif type(value) is int:
    #                 if value == 0:
    #                     self.mode = False
    #                 elif value == 1:
    #                     self.mode = True

    #             elif type(value) is str:
    #                 if value.lower() == 'parallel':
    #                     self.mode = False
    #                 elif value.lower() == 'orthographic':
    #                     self.mode = False
    #                 elif value.lower() == 'cone':
    #                     self.mode = True
    #                 elif value.lower() == 'perspective':
    #                     self.mode = True
    #             else:
    #                 self.mode = None
            
    #         # set object size
    #         elif key == 'object_size_x':
    #             self.object.size.x = value
    #         elif key == 'object_size_y':
    #             self.object.size.y = value
    #         elif key == 'object_size_z':
    #             self.object.size.z = value
            
    #         # set object spacing
    #         elif key == 'object_spacing_x':
    #             self.object.spacing.x = value
    #         elif key == 'object_spacing_y':
    #             self.object.spacing.y = value
    #         elif key == 'object_spacing_z':
    #             self.object.spacing.z = value
            
    #         # set detector size
    #         elif key == 'detector_size_u':
    #             self.detector.size.u = value
    #         elif key == 'detector_size_v':
    #             self.detector.size.v = value
        
    #         # set detector spacing
    #         elif key == 'detector_spacing_u':
    #             self.detector.spacing.u = value
    #         elif key == 'detector_spacing_v':
    #             self.detector.spacing.v = value

    #         # set distance
    #         elif key == 'distance_source2object':
    #             self.source.distance.source2object = value
    #         elif key == 'distance_source2detector':
    #             self.source.distance.source2detector = value


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

class _Motion2D():
    def __init__(self):
        self.rotation = np.array([0,])
        self.translation = np.array([[0,0],])
    def set(self, alpha, offset=None):
        if type(alpha) in [int,float]:
            alpha = np.array([alpha])
        elif type(alpha) != np.ndarray:
            alpha = np.array(alpha)
        if len(alpha.shape) != 1:
            raise ValueError('Alpha is not inappropriate.')
        self.rotation = alpha
        
        if offset is not None:
            if type(offset) != np.ndarray:
                offset = np.array(offset)
            if offset.shape == (2,):
                offset = offset[None]
            elif len(offset.shape) != 2 or offset.shape[1] != 2:
                raise ValueError('Offset is not inappropriate.')
            self.translation = offset

class _Motion():
    def __init__(self):
        self.rotation = np.array([[0,0,0],])
        self.translation = np.array([[0,0,0],])
    def set(self, alpha, beta=None, gamma=None, offset=None):
        if type(alpha) in [int, float]:
            alpha = np.array([alpha])
        elif type(alpha) != np.ndarray:
            alpha = np.array(alpha)
        if len(alpha.shape) != 1:
            raise ValueError('Alpha is not inappropriate.')
        na = len(alpha)
        if beta is None:
            beta = np.zeros(na)
        if gamma is None:
            gamma = np.zeros(na)
        self.rotation = np.stack([alpha, beta, gamma], axis=-1)

        if offset is not None:
            if type(offset) != np.ndarray:
                offset = np.array(offset)
            if offset.shape == (3,):
                offset = offset[None]
            elif len(offset.shape) !=2 or offset.shape[1] != 3:
                raise ValueError('Offset is not inappropriate.')
            self.translation = offset

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
            return True

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
        self.motion = _Motion2D()
        
class _Source():
    def __init__(self):
        self.distance = _Distance()
        self.motion = _Motion()