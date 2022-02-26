from fnmatch import translate

import data_utils as du
import tensorflow as tf
import json
import numpy as np
import random

def save_to_tfrecords(name):
    f = open("./data/annotations/"+name+".json") 
    data = json.load(f)
    writer = tf.io.TFRecordWriter("./data/"+name+'.tfrecords')
    for subject in data.keys():
        for action in data[subject].keys():
            for frame in data[subject][action].keys():
                body3D = np.array(data[subject][action][frame])
                body3D = du.to_openpose(body3D)
                body3D = du.normalize(body3D)
                azimuth = random.uniform(0, 2*np.pi)
                altitude = random.uniform(-np.pi/12, np.pi/12)
                WC = du.world_to_camera(azimuth,altitude,1)
                body3D_camera = []
                for i in range(len(body3D)-1):
                    [x,y,z] = body3D[i]
                    [xc,yc,zc,_] = np.matmul(WC,[x,y,z,1])
                    body3D_camera = np.append(body3D_camera,[xc,yc,zc])
                body3D_camera = body3D_camera.reshape(-1,3,1)
                example = to_example(body3D_camera)
                writer.write(example.SerializeToString())
    writer.close()


def to_example(body3D_openpose):
    data = {
        '2D': tf.train.Feature(float_list=tf.train.FloatList(value=body3D_openpose[:,0:2].reshape(-1))),
        'Z': tf.train.Feature(float_list=tf.train.FloatList(value=body3D_openpose[:,2]))
    }
    return tf.train.Example(features=tf.train.Features(feature=data))

save_to_tfrecords('Human36M_subject1_joint_3d')
save_to_tfrecords('Human36M_subject5_joint_3d')
save_to_tfrecords('Human36M_subject6_joint_3d')
save_to_tfrecords('Human36M_subject7_joint_3d')
save_to_tfrecords('Human36M_subject8_joint_3d')
save_to_tfrecords('Human36M_subject9_joint_3d')
save_to_tfrecords('Human36M_subject11_joint_3d')