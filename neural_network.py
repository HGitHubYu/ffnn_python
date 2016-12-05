#!/usr/bin/env python

"""Port of Huan Yu's Feed-Forward Neural Network to Python"""
# Feed-Forward Neural Network
# Modified by Jered Tupik on 11/28/2016
#
# Georgia Institute of Technology
#
# A Python-implementation of Feed-Forward Neural Networks, as demonstrated
# by Huan Yu's existing Feed-Forward Neural Network code.
import random
import sys

import matplotlib.pyplot as pyplot
import numpy

class Weight(object):
    pass

class NeuralNetwork(object):

    # Constructor of the Neural Network
    # Parameters: input  - The number of input sources/neurons. Will be a minimum of 1
    #             hidden - A varying-length list of the number of hidden neurons per layer.
    #             output - The number of output sources/neurons. Must be at least 1 in order
    #                      retrieve data.
    def __init__(self, input, hidden, output):
        self.numInput  = input
        self.numHidden = sum(hidden)
        self.numOutput = output
        self.numNodes  = input + sum(hidden) + output + 1  # +1 for Bias Node
        self.weights   = []

        i = 0
        # Initialize Weights/Connections for Input->Hidden Layer Nodes
        for s in range(0, input+1):
            for d in range(input+1, input+1+hidden[0]):
                self.weights.append(Weight())
                self.weights[i].source = s
                self.weights[i].dest   = d
                self.weights[i].val    = random.random()
                i += 1

        # Initialize Weights/Connections for all subsequent Hidden Layers
        curr_start = input+1
        if len(hidden) > 1:
            for lyr in range(0, len(hidden)-1):
                for s in ([0] + range(curr_start, curr_start+hidden[lyr])):
                    for d in range(curr_start+hidden[lyr], curr_start+hidden[lyr]+hidden[lyr+1]):
                        self.weights.append(Weight())
                        self.weights[i].source = s
                        self.weights[i].dest   = d
                        self.weights[i].val    = random.random()
                        i += 1
                curr_start += hidden[lyr]

        # Initialize Weights/Connections for Hidden Layer/Output Nodes
        for s in ([0] + range(curr_start, curr_start+hidden[len(hidden)-1])):
            for d in range(input+1+sum(hidden), input+1+sum(hidden)+output):
                self.weights.append(Weight())
                self.weights[i].source = s
                self.weights[i].dest   = d
                self.weights[i].val    = random.random()
                i += 1

    # Gives a printout of the neural network statistics, and a list of all
    # neuron connections.
    def print_info(self):
        print "Feed Forward Neural Network:"
        print "  -Inputs: %d -Hidden: %d -Outputs: %d" % (self.numInput, self.numHidden, self.numOutput)
        weights = self.weights
        for i in range(0, len(weights)):
            print "    -Weights: %.2d, %.2d, %.4f" % (weights[i].source, weights[i].dest, weights[i].val)

    """
    # Generates training data corresponding to the given type. Current allows
    # 8 different training data setups.
    # Parameters: type - The type of training data to produce
    def gen_training_data(self, type):
        training_data = [[], []]
        if type==1:     #
            training_data[0] = numpy.arange(0, 60, 60/1000.0).tolist()
            training_data[1] = numpy.sin(training_data[0])
            training_data[0] = numpy.ones([1, len(training_data[0])]).tolist()
        elif type==2:   #
            aa = numpy.arange(0, 120, 120/1000.0)
            training_data[1] = numpy.sin(aa)
            training_data[0] = numpy.ones([1, len(aa)])
        elif type==3:   #
            aa = numpy.arange(0, 120, 120/1000.0)
            bb = numpy.sin(aa)
            training_data[0] = numpy.zeros([1, 50])
            training_data[1] = numpy.zeros([1, 50])
            training_data[0] = [training_data[0], numpy.ones([1, 200])]
            training_data[1] = [training_data[1], bb[numpy.arange(1, 200+1)]]
            training_data[0] = [training_data[0], numpy.zeros([1, 60])]
            training_data[1] = [training_data[1], numpy.zeros([1, 60])]
            training_data[0] = [training_data[0], numpy.ones([1, 100])]
            training_data[1] = [training_data[1], bb[numpy.arange(1, 100+1)]]
            training_data[0] = [training_data[0], numpy.zeros([1, 30])]
            training_data[1] = [training_data[1], numpy.zeros([1, 30])]
            training_data[0] = [[training_data[0]], [0, training_data[0][numpy.arange(1, len(training_data[0]))]], [0, 0, training_data[0][numpy.arange(1, len(training_data[0])-1)]]]
            training_data[1] = [0, training_data[1][numpy.arange(1, len(training_data[1]))]]
        else:           # Default sin() waveform
            aa = numpy.arange(0, 120, 120/1000.0)
            training_data[0] = numpy.ones(1, len(aa))
            training_data[1] = numpy.sin(aa)

        print "%d, %d" % (len(training_data[0]), len(training_data[1]))
        print training_data[0]
        print training_data[1]
        #pyplot.figure()
        #pyplot.title('Training Data Input')
        #pyplot.plot(training_data[0])

        #pyplot.figure()
        #pyplot.title('Training Data Output')
        #pyplot.show(training_data[1])
        return training_data
    """

    # Returns the activation function result and derivative depending on the
    # given type. Currently allows Sigmoid and Hyperbolic Tangent Activation
    # Parameters: opt    - An integer signifying the activation function type
    #             in_sum - The double sum of all the inputs to a neuron
    def __det_act_func(self, opt, in_sum):
        act   = 0;  # Activation Function
        act_d = 0;  # Activation Function's Derivative
        
        if opt == 1:    # Sigmoid Activation Function
            act = 1 / (1 + math.exp(-in_sum))
            act_d = act * (1 - act)
        elif opt == 2:  # Hyperbolic Tangent Activation Function
            num = math.exp(in_sum) - math.exp(-in_sum)
            den = math.exp(in_sum) + math.exp(-in_sum)
            act = num / den
            act_d = 1 - (act**2)
        else:           # Default: Sigmoid Activation Function
            act = 1 / (1 + math.exp(-in_sum))
            act_d = act * (1 - act)
        return [act, act_d]

if __name__ == '__main__':
    ann = NeuralNetwork(2, [4, 4, 4], 2)
    ann.print_info()
    sys.exit
