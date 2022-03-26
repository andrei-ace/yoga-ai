import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
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
    # ax.view_init(elev=ax.elev, azim=ax.azim)
    # ax.view_init(elev=15, azim=-30)

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

batch_size=1
dataset = du.load_tfrecords2D().batch(batch_size)

CW = tf.convert_to_tensor(du.camera_to_world(),dtype=np.float32)

model_fname = './model/gan/gan.h5'
generator = tf.keras.models.load_model(model_fname)
generator.summary()

for X in dataset.take(1):
    Y = generator(X,training=False)
    X = tf.reshape(X,(-1, 14, 2))
    ones = tf.ones(Y.shape);
    body = tf.stack([X[:,:,0],X[:,:,1], Y, ones], axis=2)
    body_3d = tf.transpose(tf.matmul(CW, body,transpose_a=False,transpose_b=True),perm=[0,2,1])
    draw_fig_2D_openpose(body[0,:,:-1], 'gan2D.png')
    draw_fig_3D_openpose(body_3d[0,:,:-1], 'gan3D.png')