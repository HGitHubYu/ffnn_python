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

    # Simulates one walkthrough of the Feed Forward Neural Network. Requires
    # that the input be the same dimensions as the number of input neurons.
    # Parameters: input   - Input Data for the neural network.
    def simulate(self, input):

        # Get the size of the input, and convert it to a numpy array, if not
        # already.
        input = numpy.array(input)
        input_dim = len(input)
        if input.ndim == 2:
            input_length = len(input[0])
        else:
            input_length = input_dim
            input_dim = 1

        if input_dim != self.numInput:
            print "Number of Input Units and Input Patterns do not match."
            return

        

        # Setting Parameters
        first_step = 0
        last_step = input_length

        ACT = numpy.zeros((self.numNodes, last_step))
        ACT[0, :] = 1;  # Constant Bias Unit ACT(1,:)
        ACT[1:1+self.numInput, first_step:last_step+1] = input  # ACT(2:numInput+1, firstStep:lastStep)
        ACTD = numpy.zeros((self.numNodes, last_step))

        print ACT
        print ACTD

if __name__ == '__main__':
    ann = NeuralNetwork(2, [4, 4, 4], 2)
    #ann.print_info()
    ann.simulate([[1, 0.5], [2, 1.5]])
    sys.exit
