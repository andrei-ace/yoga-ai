import data_utils as du
import tensorflow as tf
import json
import numpy as np
import glob
import os
import sys
from pathlib import Path

def save_to_tfrecords_from2D(name):
    f = open(name) 
    data = json.load(f)    
    writer = tf.io.TFRecordWriter("./data/gan/"+Path(name).stem+'.tfrecords')
    for point in data:
        point = np.array(point)
        body2D = np.append(point,[(point[8][0]+point[11][0])/2,(point[8][1]+point[11][1])/2])
        body2D = body2D.reshape(-1,2)
        body2D = du.normalize2D(body2D)

        IC = du.image_to_camera()
        body2D_camera = []
        for i in range(len(body2D)):
            [x,y] = body2D[i]
            [xc,yc,_] = np.matmul(IC,[x,y,1])
            body2D_camera = np.append(body2D_camera,[xc,yc])
        body2D_camera = body2D_camera.reshape(body2D.shape)
        example = to_example(body2D_camera[:-1])
        writer.write(example.SerializeToString())
    writer.close()

def to_example(body2D_openpose):
    x = body2D_openpose.flatten()
    return tf.train.Example(features=tf.train.Features(feature={
        "x": tf.train.Feature(float_list=tf.train.FloatList(value=x))
    }))


for json_file in glob.glob(os.path.join(sys.argv[1],"*.json")):
    save_to_tfrecords_from2D(json_file)