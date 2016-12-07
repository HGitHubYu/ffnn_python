#!/usr/bin/env python

"""Port of Huan Yu's Feed-Forward Neural Network to Python"""
# Feed-Forward Neural Network
# Modified by Jered Tupik on 12/6/2016
#
# Georgia Institute of Technology
#
# A Python-implementation of Feed-Forward Neural Networks, as demonstrated
# by Huan Yu's existing Feed-Forward Neural Network code.
import math
import random
import sys

import matplotlib.pyplot as pyplot
import numpy

from training_generation import gen_training_data

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
        ACT[0, :] = 1  # Constant Bias Unit ACT(1,:)
        ACT[1:1+self.numInput, first_step:last_step+1] = input  # ACT(2:numInput+1, firstStep:lastStep)
        ACTD = numpy.zeros((self.numNodes, last_step))

        # Assign Parameters
        weights_source = []
        weights_dest = []
        weights_val = []
        for i in range(0, len(self.weights)):
            weight = self.weights[i]
            weights_source.append(weight.source)
            weights_dest.append(weight.dest)
            weights_val.append(weight.val)
        weights_dest.append(-1) # Used as a sign if the index goes out of range

        for step in range(first_step, last_step):

            # Feed Forward
            next_dest = weights_dest[0]
            weight_index = 0;

            while weight_index < len(self.weights):
                unit_input_sum = 0
                dest = next_dest
                while dest == next_dest:         
                    unit_input_sum = unit_input_sum + (weights_val[weight_index] * ACT[weights_source[weight_index], step])
                    weight_index = weight_index + 1
                    next_dest = weights_dest[weight_index]
                if dest >= (self.numNodes - self.numOutput):
                    # Output Unit, Derivative = 1
                    ACT[dest, step] = unit_input_sum;
                    ACTD[dest, step] = 1;
                else:
                    # Hidden Unit, Use Activation Function
                    ACT[dest, step], ACTD[dest, step] = self.__det_act_func(2, unit_input_sum)

        output = ACT[(self.numNodes - self.numOutput):(self.numNodes+1), first_step:(last_step+1)]
        return output

    # Trains our Feed Forward Neural Network using back propogation.
    # Requires that our input and output training data conforms to the
    # number of input and output neurons.
    # Parameters: train_in  - The input data to be trained on.
    #             train_out - The output data to be trained on.
    #             d_weight  - The maximum weight change per interation.
    def train(self, train_in, train_out, d_weight):

        # Parameter Checking
        # Get the size of the input, and convert it to a numpy array, if not
        # already.
        train_in = numpy.array(train_in)
        input_dim = len(train_in)
        if train_in.ndim == 2:
            input_length = len(train_in[0])
        else:
            input_length = input_dim
            input_dim = 1

        # Get the size of the output, and convert it to a numpy array, if not
        # already.
        train_out = numpy.array(train_out)
        output_dim = len(train_out)
        if train_out.ndim == 2:
            output_length = len(train_out[0])
        else:
            output_length = output_dim
            output_dim = 1

        if input_dim != self.numInput:
            print "Number of Input Units and Input Patterns do not match."
            return
        if output_dim != self.numOutput:
            print "Number of Output Units and Target Patterns do not match."
            return
        if input_length != output_length:
            print "Length of Input and Output Samples is Different."
            return

        # Setting Parameters
        first_step = 0
        last_step = input_length

        ACT = numpy.zeros((self.numNodes, last_step))
        ACT[0, :] = 1   # Constant Bias Unit
        ACT[1:1+self.numInput, first_step:last_step+1] = train_in
        ACTD = numpy.zeros((self.numNodes, last_step))

        # Assign Parameters
        # Assign Parameters
        weights_source = []
        weights_dest = []
        weights_val = []
        for i in range(0, len(self.weights)):
            weight = self.weights[i]
            weights_source.append(weight.source)
            weights_dest.append(weight.dest)
            weights_val.append(weight.val)
        weights_dest.append(-1) # Used as a sign if the index goes out of range

        step_output_error_derivatives_weights = numpy.zeros((len(weights_source), self.numOutput, last_step))
        total_output_error_derivatives_weights = numpy.zeros(len(weights_source))

        # Main Loop
        for step in range(first_step, last_step):

            # Feed Forward
            next_dest = weights_dest[0]
            weight_index = 0
            while weight_index < len(weights_source):
                unit_input_sum = 0;
                dest = next_dest
                while dest == next_dest:
                    unit_input_sum = unit_input_sum + (weights_val[weight_index] * ACT[weights_source[weight_index], step])
                    weight_index = weight_index + 1
                    next_dest = weights_dest[weight_index]
                if dest >= (self.numNodes - self.numOutput):
                    # Output Unit, Derivative = 1
                    ACT[dest, step] = unit_input_sum
                    ACTD[dest, step] = 1
                else:
                    # Hidden Unit, Use Activation Function
                    ACT[dest, step], ACTD[dest, step] = self.__det_act_func(2, unit_input_sum)

            # Back Propogation
            output_derivatives_weights = numpy.zeros((len(weights_source), self.numOutput))
            output_derivatives_unit_activity = numpy.zeros((self.numNodes, self.numOutput))
            output_derivatives_unit_activity[(self.numNodes-self.numOutput):self.numNodes, 0:self.numOutput] = numpy.identity(self.numOutput)

            next_dest = weights_dest[len(weights_source)]
            weight_index = len(weights_source)-1
            while weight_index >= 0:
                dest = next_dest
                while dest == next_dest:
                    source = weights_source[weight_index]
                    output_derivatives_weights[weight_index, :] = numpy.multiply(output_derivatives_unit_activity[dest, :], numpy.multiply(ACTD[dest, step], ACT[source, step]))

                    # Calculate derivatives
                    output_derivatives_unit_activity[source, :] = numpy.add(output_derivatives_unit_activity[source, :], numpy.multiply(output_derivatives_unit_activity[dest, :], numpy.multiply(ACTD[dest, step], weights_val[weight_index])))

                    # Get Next Destination Node
                    weight_index = weight_index - 1
                    if weight_index < 0:
                        break;
                    next_dest = weights_dest[weight_index]

            # Calculate the current step output error derivatives
            for UI in range(0, self.numOutput):
                if train_out.ndim == 1:
                    step_output_error_derivatives_weights[:, UI, step] = numpy.multiply(output_derivatives_weights[:, UI], (ACT[UI+self.numNodes-self.numOutput, step] - train_out[step]))
                else:
                    step_output_error_derivatives_weights[:, UI, step] = numpy.multiply(output_derivatives_weights[:, UI], (ACT[UI+self.numNodes-self.numOutput, step] - train_out[UI, step]))

        # Calculate the total output error derivatives
        for WI in range(0, len(weights_source)):
            for UI in range(0, self.numOutput):
                for st in range(first_step, last_step):
                    total_output_error_derivatives_weights[WI] = (total_output_error_derivatives_weights[WI]+step_output_error_derivatives_weights[WI, UI, st])

        # Adjust Weights
        weights_val = numpy.subtract(weights_val, numpy.multiply(total_output_error_derivatives_weights, d_weight))
        for WI in range(1, len(weights_source)):
            self.weights[WI].val = weights_val[WI];

    
if __name__ == '__main__':
    #ann = NeuralNetwork(10, [4, 4, 4], 4)
    #ann.print_info()
    #for i in range(0, 10):
    #    print "Simulation Run: %d. Target (1, 2, 3, 4): " % i
    #    print ann.simulate([[0], [0.1], [0.2], [0.3], [0.4], [0.5], [0.7], [0.8], [0.9], [1.0]])
    #    ann.train([[0], [0.1], [0.2], [0.3], [0.4], [0.5], [0.7], [0.8], [0.9], [1.0]], [[1], [2], [3], [4]], 1)
    train_in, train_out = gen_training_data(7)

    input_dim = len(train_in)
    if train_in.ndim == 2:
        input_length = len(train_in[0])
    else:
        input_length = input_dim
        input_dim = 1

    output_dim = len(train_out)
    if train_out.ndim == 2:
        output_length = len(train_out[0])
    else:
        output_length = output_dim
        output_dim = 1

    num_input = input_dim
    hidden_neurons = [5]
    num_output = output_dim
    d_weight = 0.0005#0.0002
    error = []
    error_threshold = 0.05#0.005
    error_iteration = 1000000
    training_iteration=0
    training_iteration_threshold=500#1000

    ann = NeuralNetwork(num_input, hidden_neurons, num_output)

    while (error_iteration > error_threshold) and (training_iteration < training_iteration_threshold):
        error_iteration = 0
        training_iteration = training_iteration + 1
        ann.train(train_in, train_out, d_weight)
        output = ann.simulate(train_in)
        for m in range(0, output_dim):
            for n in range(0, output_length):
                if output_dim == 1:
                    error_iteration = error_iteration + 0.5 * ((output[m, n] - train_out[n])**2)
                else:
                    error_iteration = error_iteration + 0.5 * ((output[m, n] - train_out[m, n])**2)
        error.append(error_iteration)
        print "%d, %f" % (training_iteration, error_iteration)

    print "Iterations Taken: %d" % training_iteration
    print "Final Error: %f" % error_iteration
   
    pyplot.figure()
    pyplot.subplot(1, 2, 1)
    pyplot.title("Neural Network Error")
    pyplot.plot(error)
    pyplot.subplot(1, 2, 2)
    pyplot.title("Input/Output Comparisons")
    pyplot.plot(train_out, 'b-')
    pyplot.plot(output[0], 'r-')
    #pyplot.plot(train_in[1, :], 'g-')
    pyplot.show()

    sys.exit
