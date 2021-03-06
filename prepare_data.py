import data_utils as du
import tensorflow as tf
import json
import numpy as np
import random

MULTIPLY = 5


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
                altitude = 0
                WC = du.world_to_camera(azimuth, altitude)
                body3D_camera = []
                for i in range(len(body3D)-1):
                    [x, y, z] = body3D[i]
                    [xc, yc, zc, _] = np.matmul(WC, [x, y, z, 1])
                    body3D_camera = np.append(body3D_camera, [xc, yc, zc])
                body3D_camera = body3D_camera.reshape(-1, 3)
                example = to_example(body3D_camera)
                writer.write(example.SerializeToString())
    writer.close()


def to_example(body3D_openpose):
    x = body3D_openpose[:, 0:2].flatten()
    y = body3D_openpose[:, 2].flatten()
    return tf.train.Example(features=tf.train.Features(feature={
        "x": tf.train.Feature(float_list=tf.train.FloatList(value=x)),
        "y": tf.train.Feature(float_list=tf.train.FloatList(value=y)),
    }))


save_to_tfrecords('Human36M_subject1_joint_3d')
save_to_tfrecords('Human36M_subject5_joint_3d')
save_to_tfrecords('Human36M_subject6_joint_3d')
save_to_tfrecords('Human36M_subject7_joint_3d')
save_to_tfrecords('Human36M_subject8_joint_3d')
save_to_tfrecords('Human36M_subject9_joint_3d')
save_to_tfrecords('Human36M_subject11_joint_3d')
