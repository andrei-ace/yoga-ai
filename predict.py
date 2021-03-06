import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
import matplotlib.pyplot as plt
import data_utils as du

def draw_fig_2D_openpose(body3D_openpose, name):
    body3D = np.zeros(shape=(15,3))
    body3D[:-1] = body3D_openpose 
    body3D[14] = [(body3D[8][0]+body3D[11][0])/2,(body3D[8][1]+body3D[11][1])/2,(body3D[8][2]+body3D[11][2])/2]
    
    fig = plt.figure()
    start = [14, 8, 9,14,11,12,14, 1, 1, 2, 3, 1, 5, 6]
    end   = [ 8, 9,10,11,12,13, 1, 0, 2, 3, 4, 5, 6, 7]
    colors= [ 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]

    rcolor='blue'
    lcolor='red'
    for i in range(len(start)):
        x = [body3D[start[i],0],body3D[end[i],0]]
        y = [body3D[start[i],1],body3D[end[i],1]]
        plt.plot(x,y, c = lcolor if colors[i] else rcolor)

    plt.xlim([1.2, -1.2])
    plt.ylim([-1.2, 1.2])

    plt.xlabel("-x")
    plt.ylabel("y")

    plt.savefig(name)


def draw_fig_3D_openpose(body3D_openpose, name):
    body3D = np.zeros(shape=(15,3))
    body3D[:-1] = body3D_openpose 
    body3D[14] = [(body3D[8][0]+body3D[11][0])/2,(body3D[8][1]+body3D[11][1])/2,(body3D[8][2]+body3D[11][2])/2]
    fig = plt.figure()
    ax = plt.axes(projection ='3d')

    start = [14, 8, 9,14,11,12,14, 1, 1, 2, 3, 1, 5, 6]
    end   = [ 8, 9,10,11,12,13, 1, 0, 2, 3, 4, 5, 6, 7]
    colors= [ 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]

    rcolor='blue'
    lcolor='red'
    for i in range(len(start)):
        x = [body3D[start[i],0],body3D[end[i],0]]
        y = [body3D[start[i],1],body3D[end[i],1]]
        z = [body3D[start[i],2],body3D[end[i],2]]
        ax.plot(x,y,z, c = lcolor if colors[i] else rcolor)

    ax.set_xlim3d([-1.2,1.2])
    ax.set_ylim3d([-1.2,1.2])
    ax.set_zlim3d([-1.2,1.2])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.savefig(name)


batch_size = 64
dataset = du.load_tfrecords().shuffle(1000*batch_size, reshuffle_each_iteration=True)
dataset = dataset.batch(batch_size)

model = tf.keras.models.load_model('./model/article/article.h5')
X,Y = list(dataset.take(8).as_numpy_iterator())[0]
X=X[0]
Y=Y[0]

body3D_camera = np.zeros((14,3))
body3D_camera[:,:2] = X.reshape(14,2)
body3D_camera[:,2] = Y

CW = du.camera_to_world()
body3D = []
for i in range(len(body3D_camera)):
    [x,y,z] = body3D_camera[i]
    [xc,yc,zc,_] = np.matmul(CW,[x,y,z,1])
    body3D = np.append(body3D,[xc,yc,zc])
body3D = body3D.reshape(body3D_camera.shape)

draw_fig_2D_openpose(body3D_camera, 'real2D.png')
draw_fig_3D_openpose(body3D, 'real3D.png')

Y = model.predict(X.reshape(1,28))

body3D_camera = np.zeros((14,3))
body3D_camera[:,:2] = X.reshape(14,2)
body3D_camera[:,2] = Y

CW = du.camera_to_world()
body3D = []
for i in range(len(body3D_camera)):
    [x,y,z] = body3D_camera[i]
    [xc,yc,zc,_] = np.matmul(CW,[x,y,z,1])
    body3D = np.append(body3D,[xc,yc,zc])
body3D = body3D.reshape(body3D_camera.shape)
draw_fig_3D_openpose(body3D, 'predicted3D.png')
