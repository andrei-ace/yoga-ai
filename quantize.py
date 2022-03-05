import argparse
import os
import shutil
import sys

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_quantize
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from data_utils import load_tfrecords

DIVIDER = '-----------------------------------------'



def quant_model(float_model,quant_model,batchsize,evaluate):
    '''
    Quantize the floating-point model
    Save to HDF5 file
    '''

    # make folder for saving quantized model
    head_tail = os.path.split(quant_model) 
    os.makedirs(head_tail[0], exist_ok = True)

    # load the floating point trained model
    float_model = load_model(float_model)
    float_model.summary()

    quant_dataset = load_tfrecords().batch(batchsize, drop_remainder=False)
    
    # run quantization
    quantizer = vitis_quantize.VitisQuantizer(float_model)
    quantized_model = quantizer.quantize_model(calib_dataset=quant_dataset)

    # saved quantized model
    quantized_model.save(quant_model)
    print('Saved quantized model to',quant_model)


    if (evaluate):
        '''
        Evaluate quantized model
        '''
        print('\n'+DIVIDER)
        print ('Evaluating float model..')
        print(DIVIDER+'\n')

        test_dataset = load_tfrecords().batch(batchsize, drop_remainder=False)

        float_model.compile(optimizer=Adam(),
                                loss='mse',
                                metrics=['mse'])

        scores = float_model.evaluate(test_dataset,
                                          verbose=0)

        print('Float model mse:',scores[1])
        print('\n'+DIVIDER)
        print ('Evaluating quantized model..')
        print(DIVIDER+'\n')

        test_dataset = load_tfrecords().batch(batchsize, drop_remainder=False)

        quantized_model.compile(optimizer=Adam(),
                                loss='mse',
                                metrics=['mse'])

        scores = quantized_model.evaluate(test_dataset,
                                          verbose=0)

        print('Quantized model mse:',scores[1])
        print('\n'+DIVIDER)

    return



def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--float_model',  type=str, default='build/float_model/f_model.h5', help='Full path of floating-point model. Default is build/float_model/k_model.h5')
    ap.add_argument('-q', '--quant_model',  type=str, default='build/quant_model/q_model.h5', help='Full path of quantized model. Default is build/quant_model/q_model.h5')
    ap.add_argument('-b', '--batchsize',    type=int, default=50,                       help='Batchsize for quantization. Default is 50')
    ap.add_argument('-e', '--evaluate',     action='store_true', help='Evaluate floating-point model if set. Default is no evaluation.')
    args = ap.parse_args()  

    print('\n------------------------------------')
    print('TensorFlow version : ',tf.__version__)
    print(sys.version)
    print('------------------------------------')
    print ('Command line options:')
    print (' --float_model  : ', args.float_model)
    print (' --quant_model  : ', args.quant_model)
    print (' --batchsize    : ', args.batchsize)
    print (' --evaluate     : ', args.evaluate)
    print('------------------------------------\n')


    quant_model(args.float_model, args.quant_model, args.batchsize, args.evaluate)


if __name__ ==  "__main__":
    main()