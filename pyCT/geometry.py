import pyCT
from pyCT.parameter import _Parameters
import numpy as np
from skimage.transform import rescale
import matplotlib.pyplot as plt
from matplotlib import cm

def show(params : _Parameters, 
         angle = 0, 
         offset = [0,0,0],
         obj = None, 
         scale = 4,
         view = None,
         *args):
    
    ax = plt.figure(figsize=(10,10)).add_subplot(projection='3d')
    
    params2 = params.copy()
    params2.object.size.set(np.array(params.object.size.get())//scale)
    params2.object.spacing.set(np.array(params.object.spacing.get())*scale)
    params2.set()
    tf = pyCT.forward.getTransformation(params2)
    _ = tf.getMatrix(angle, offset, *args)

    if obj is not None:
        # volume
        cube = rescale(obj, 1/scale, preserve_range=True, anti_aliasing=False)
        z, y, x = np.indices(np.array(cube.shape)+1)
        x, y, z, _ = tf.worldTransformation[0] @ np.stack([x.flatten(), y.flatten(), z.flatten(),np.ones_like(z.flatten())])
        x = x.reshape(params2.object.size.z+1, params2.object.size.y+1, params2.object.size.x+1) - params2.object.spacing.x/2
        y = y.reshape(params2.object.size.z+1, params2.object.size.y+1, params2.object.size.x+1) - params2.object.spacing.y/2
        z = z.reshape(params2.object.size.z+1, params2.object.size.y+1, params2.object.size.x+1) - params2.object.spacing.z/2
        filled = cube>0
        facecolors = cm.viridis(cube)
        facecolors[...,-1] = 1-cube
        ax.voxels(x, y, z, filled=filled, facecolors=facecolors, shade=True)

        # projection
        proj = pyCT.forward.project(obj, params, angles=angle, offsets=offset)[0]
        su, sv = params.detector.length.get()
        nu, nv = params.detector.size.get()
        X,Y = np.meshgrid(np.linspace(.5,su-.5,nu)-su/2, np.linspace(.5,sv-.5,nv)-sv/2)
        Z = np.ones_like(Y) * -params.distance.source2detector
        X,Y,Z,_ = np.linalg.inv(tf.cameraTransformation[0]) @ np.array([X.flatten(), Y.flatten(), Z.flatten(), np.ones(X.size)])
        X = X.reshape(proj.shape)
        Y = Y.reshape(proj.shape)
        Z = Z.reshape(proj.shape)
        ax.plot_surface(X, Y, Z, facecolors=np.repeat(proj[...,None], axis=-1, repeats=3)/proj.max(), alpha=.3)
        
    # volume outline
    x, y, z = params.object.length.get()
    ax.plot([-x/2,-x/2,x/2,x/2,-x/2], [-y/2,y/2,y/2,-y/2,-y/2], [z/2,z/2,z/2,z/2,z/2], '--', color='gray', alpha=.5)
    ax.plot([-x/2,-x/2,x/2,x/2,-x/2], [-y/2,y/2,y/2,-y/2,-y/2], [-z/2,-z/2,-z/2,-z/2,-z/2], '--', color='gray', alpha=.5)
    ax.plot([-x/2,-x/2],[-y/2,-y/2],[-z/2,z/2], '--', color='gray', alpha=.5)
    ax.plot([x/2,x/2],[-y/2,-y/2],[-z/2,z/2], '--', color='gray', alpha=.5)
    ax.plot([-x/2,-x/2],[y/2,y/2],[-z/2,z/2], '--', color='gray', alpha=.5)
    ax.plot([x/2,x/2],[y/2,y/2],[-z/2,z/2], '--', color='gray', alpha=.5)
    
    # axis
    sx, sy, sz = params.object.length.get()
    ax.plot([-sx,sx],[0,0],[0,0], 'k')
    ax.text(sx,0,0,'x')
    ax.plot([0,0],[-sy,sy],[0,0], 'k')
    ax.text(0,sy,0,'y')
    ax.plot([0,0],[0,0],[-sz,sz], 'k')
    ax.text(0,0,sz,'z')

    # camera
    cam = np.linalg.inv(tf.cameraTransformation)
    right = cam[0,:-1,0]
    up = cam[0,:-1,1]
    back = cam[0,:-1,2]
    source = cam[0,:-1,3]
    detector_center = source - back * params.distance.source2detector
    detector_right = detector_center + right*params.detector.length.u/2
    detector_up = detector_center + up*params.detector.length.v/2
    detector_right_up = detector_center + right*params.detector.length.u/2 + up*params.detector.length.v/2
    detector_left_up = detector_center - right*params.detector.length.u/2 + up*params.detector.length.v/2
    detector_right_down = detector_center + right*params.detector.length.u/2 - up*params.detector.length.v/2
    detector_left_down = detector_center - right*params.detector.length.u/2 - up*params.detector.length.v/2
    
    temp = np.stack([source, detector_center]).T
    ax.plot(temp[0],temp[1],temp[2],color='orange',marker='o',linestyle='dashed')
    temp = np.stack([detector_right_up, detector_right_down, detector_left_down, detector_left_up, detector_right_up]).T
    ax.plot(temp[0],temp[1],temp[2],color='red')
    temp = np.stack([detector_center, detector_right]).T
    ax.plot(temp[0], temp[1], temp[2], 'r')
    ax.text(detector_right[0], detector_right[1], detector_right[2], 'u')
    temp = np.stack([detector_center, detector_up]).T
    ax.plot(temp[0], temp[1], temp[2], 'r')
    ax.text(detector_up[0], detector_up[1], detector_up[2], 'v')
    
    ax.axis('equal')
    if view is not None:
        ax.view_init(azim=view[0], elev=view[1], roll=view[2])