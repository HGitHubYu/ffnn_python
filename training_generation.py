#!/usr/bin/env python

"""Port of Huan Yu's Feed-Forward Neural Network to Python"""
# Training Data Generator
# Modified by Jered Tupik on 11/28/2016
#
# Georgia Institute of Technology
#
# A Python-implementation of Feed-Forward Neural Networks, as demonstrated
# by Huan Yu's existing Feed-Forward Neural Network code. Acts as a generator
# for training data in the neural network.
import sys

import matplotlib.pyplot as pyplot
import numpy

# Generates training data corresponding to the given type. Current allows
# 8 different training data setups.
# Parameters: opt - The type of training data to produce
def gen_training_data(opt):
    train_in  = []
    train_out = []
    
    if opt == 1:
        train_in = None
    elif opt == 2:
        train_in = None
    elif opt == 3:
        train_in = None
    elif opt == 4:
        #aa = numpy.zeros(10) + numpy.ones(10) + numpy.zeros(10) + numpy.ones(10) + numpy.zeros(10) + numpy.ones(10)
        #bb = aa + aa + aa + aa + aa + aa + aa + aa + aa + aa + aa + aa
        #train_in = numpy.zeros(90) + numpy.ones(80) + numpy.zeros(80) + numpy.ones(60) + numpy.zeros(100) + numpy.ones(100) + numpy.zeros(50)
        #input_length = length(train_in)
        #train_out = numpy.multiply(train_in, bb[:input_length])
        #train_in = [[train_in], [0, train_in[:len(train_in)-1], [0, 0, train_in[:len(train_in)-2]]]]
        train_in = None
    elif opt == 5:
        aa = numpy.arange(0, 50+0.1, 0.1)   #0:0.1:50
        bb = numpy.sin(aa)
        cc = [0] + bb
        dd = numpy.subtract(numpy.power(bb, 2), numpy.multiply(cc, 2))
        train_out = numpy.zeros(len(dd))
        train_out[0] = dd[0]
        for n in range(1, len(dd)):
            train_out[n] = dd[n] - train_out[n-1]
        train_in = [[bb], [cc]]
    elif opt == 6:
        aa = numpy.multiply(numpy.random.random(1000), 2)
        bb = numpy.multiply(numpy.random.random(1000), 2)
        cc = numpy.multiply(aa, bb)
        dd = numpy.random.random(1000)
        
        train_out = numpy.multiply(numpy.subtract(cc, numpy.power(numpy.subtract(aa, bb), 2)), dd)
        train_in = [[aa], [bb], [cc], [dd]]
    elif opt == 7:
        aa = numpy.arange(0, 1.5001, 0.001)  # 0:0.001:1.5
        bb = numpy.arange(0, 1.5001, 0.001)  # 0:0.001:1.5
        cc = numpy.multiply(aa, bb)
        
        train_out = numpy.subtract(cc, numpy.power(numpy.subtract(aa, bb), 2))
        train_in = [[aa], [bb], [cc]]
    elif opt == 8:
        aa = numpy.arange(0, 60, 0.1) # 0:0.1:60
        bb = numpy.sin(aa)
        lg = len(aa)
        bb_d = numpy.zeros(lg)
        for n in numpy.arange(1, lg):
            bb_d[n] = bb[n] - bb[n-1]
            
        train_in = [[bb[2:len(bb)-1]], [bb_d[2:len(bb_d)-1]]]
        train_out = bb_d[3:]
    else:           # Default sin() waveform
        aa = numpy.arange(0, 120+120/1000.0, 120/1000.0)
        
        train_out = numpy.sin(aa)
        train_in = numpy.ones(len(train_out))

    # Graph the Training Data Inputs and Outputs
    pyplot.figure()
    #pyplot.subplot(2, 1, 1)
    #pyplot.title("Training Data Input")
    #pyplot.plot(train_in)
    pyplot.subplot(2, 1, 2)
    pyplot.title("Training Data Output")
    pyplot.plot(train_out)
    pyplot.show()
    
    return [train_in, train_out]


if __name__ == '__main__':
    for i in range(5, 10):
        gen_training_data(i)
    sys.exit
