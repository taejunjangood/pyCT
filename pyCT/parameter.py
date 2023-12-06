import yaml

def getParameters(path_header):
    return _Parameters(path_header)

class _Parameters():
    def __init__(self, path_header):
        # set Mode
        self.mode = None
        # set Object
        self.object = Struct(size = Struct(x=None, y=None, z=None), 
                             spacing = Struct(x=None, y=None, z=None), 
                             length  = Struct(x=None, y=None, z=None), 
                             offset  = Struct(x=None, y=None, z=None),
                             rotation = Struct(azim=None, elev=None))
        self.detector = Struct(size = Struct(u=None, v=None), 
                               spacing = Struct(u=None, v=None), 
                               length  = Struct(u=None, v=None), 
                               offset  = Struct(u=None, v=None))
        self.distance = Struct(source2object=None, source2detector=None, near=None, far=None)
        
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
        output +=     'source to detector (mm) : {}\n'.format(self.distance.source2object)
        output +=     'source to detector (mm) : {}\n'.format(self.distance.source2detector)
        return output
        
        
    
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
            
            # set object offsets
            elif key == 'object_offset_x':
                self.object.offset.x = value
            elif key == 'object_offset_y':
                self.object.offset.y = value
            elif key == 'object_offset_z':
                self.object.offset.z = value
            
            # set object rotations
            elif key == 'object_rotation_azimuth':
                self.object.rotation.azim = value
            elif key == 'object_rotation_elevation':                
                self.object.rotation.elev = value

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
        
            # set object offset
            elif key == 'detector_offset_u':
                self.detector.offset.u = value
            elif key == 'detector_offset_v':
                self.detector.offset.v = value

            # set distance
            elif key == 'distance_source2object':
                self.distance.source2object = value
            elif key == 'distance_source2detector':
                self.distance.source2detector = value

class Struct():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)