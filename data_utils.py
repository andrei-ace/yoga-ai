import json
import numpy as np
from math import pi
import tensorflow as tf
import glob

H36M_NAMES = ['']*17
H36M_NAMES[0]  = 'Hip'
H36M_NAMES[1]  = 'RHip'
H36M_NAMES[2]  = 'RKnee'
H36M_NAMES[3]  = 'RFoot'
H36M_NAMES[4]  = 'LHip'
H36M_NAMES[5]  = 'LKnee'
H36M_NAMES[6]  = 'LFoot'
H36M_NAMES[7]  = 'Spine'
H36M_NAMES[8]  = 'Thorax'
H36M_NAMES[9]  = 'Neck'
H36M_NAMES[10] = 'Head'
H36M_NAMES[11] = 'LShoulder'
H36M_NAMES[12] = 'LElbow'
H36M_NAMES[13] = 'LWrist'
H36M_NAMES[14] = 'RShoulder'
H36M_NAMES[15] = 'RElbow'
H36M_NAMES[16] = 'RWrist'


# 0: head, 1: neck, 2: L_shoulder, 3:L_elbow, 4: L_wrist, 5: R_shoulder, 6: R_elbow, 7: R_wrist, 
# 8: L_hip, 9:L_knee, 10: L_ankle, 11: R_hip, 12: R_knee, 13: R_ankle 
OPENPOSE_NAMES = ['']*15
OPENPOSE_NAMES[0]  = 'Head'
OPENPOSE_NAMES[1]  = 'Neck'
OPENPOSE_NAMES[2]  = 'LShoulder'
OPENPOSE_NAMES[3]  = 'LElbow'
OPENPOSE_NAMES[4]  = 'LWrist'
OPENPOSE_NAMES[5]  = 'RShoulder'
OPENPOSE_NAMES[6]  = 'RElbow'
OPENPOSE_NAMES[7]  = 'RWrist'
OPENPOSE_NAMES[8]  = 'LHip'
OPENPOSE_NAMES[9] = 'LKnee'
OPENPOSE_NAMES[10] = 'LFoot'
OPENPOSE_NAMES[11] = 'RHip'
OPENPOSE_NAMES[12] = 'RKnee'
OPENPOSE_NAMES[13] = 'RFoot'
OPENPOSE_NAMES[14]  = 'Hip' #root node, computed

def normalize(body3D):
    scale_hip_head = 1./length(body3D[14], body3D[1])
    T = np.matmul(scale(scale_hip_head),translate(body3D[14]))
    centered = []
    for i in range(len(body3D)):
        point = np.matmul(T, np.append(body3D[i], 1))
        centered = np.append(centered,point[0:3])
    return centered.reshape((-1,3,))

def translate(p):
    return np.array([
        [1,0,0, -p[0]],
        [0,1,0, -p[1]],
        [0,0,1, -p[2]],
        [0,0,0,   1]])

def scale(s):
    return np.array([
        [s,0,0,0],
        [0,s,0,0],
        [0,0,s,0],
        [0,0,0,1]])


def length(p1,p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2)


def world_to_camera(rz, ry, d):
    RC = np.array([
        [ 0,-1, 0],
        [ 0, 0, 1],
        [-1, 0, 0]
    ])

    RY = np.array([
            [ np.cos(ry),  0,np.sin(ry)],
            [          0,  1,         0],
            [-np.sin(ry),  0,np.cos(ry)]
        ])

    RZ = np.array([
                [np.cos(rz), -np.sin(rz),0],
                [np.sin(rz),  np.cos(rz),0],
                [         0,           0,1]
            ])
                
    ROTC = np.matmul(RC,np.matmul(RY,RZ)) 
    TRANSF = np.zeros((4, 4))
    TRANSF[:3,:3] = ROTC
    TRANSF[2,3] = -d
    TRANSF[3,3] = 1
    return TRANSF

def camera_to_world(d):
    RC = np.array([
        [ 0,-1, 0],
        [ 0, 0, 1],
        [-1, 0, 0]
    ])

    TRANSF = np.zeros((4, 4))
    TRANSF[:3,:3] = RC.T
    TRANSF[0,3] = -d
    TRANSF[3,3] = 1
    return TRANSF


def to_openpose(bodyH3GG):
    body3d_openpose = [
        bodyH3GG[10],
        bodyH3GG[9],
        bodyH3GG[11],
        bodyH3GG[12],
        bodyH3GG[13],
        bodyH3GG[14],
        bodyH3GG[15],
        bodyH3GG[16],
        bodyH3GG[4],
        bodyH3GG[5],
        bodyH3GG[6],
        bodyH3GG[1],
        bodyH3GG[2],
        bodyH3GG[3],
        bodyH3GG[0]
    ]
    return body3d_openpose

def parse_example(example):
    data = {
        '2D': tf.io.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
        'Z': tf.io.FixedLenSequenceFeature([],tf.float32,allow_missing=True)
    }
    content = tf.io.parse_single_example(example, data)
    return content['2D'], content['Z']


def load_tfrecords():
    dataset = tf.data.TFRecordDataset(glob.glob('./data/Human36M_*.tfrecords'))
    dataset = dataset.map(parse_example).map(lambda x,y: (tf.reshape(x,(-1,28)),tf.reshape(y,(-1,14))))
    return dataset
