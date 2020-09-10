# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:24:42 2020

@author: ewand
"""


import numpy as np


class NeuralNetwork:
  def __init__(self, x, y, epochs, width, n_hidden, eta, lam):
    self.input = x
    self.y = y
    self.width = width
    self.n_hidden = n_hidden
    input_weights = [np.random.normal(size = (self.input.shape[1],self.width))]
    hidden_weights = [np.random.normal(size = (self.width, self.width)) for i in range(self.n_hidden - 1)] #Need depth - 1 weight matrices between input and output
    output_weights = [np.random.normal(size = (self.width,self.y.shape[1]))]         
    self.weights = input_weights + hidden_weights + output_weights
    self.shapes = [np.shape(arr) for arr in self.weights]
    self.output = np.zeros(self.y.shape)
    self.epochs = epochs
    self.eta = eta
    self.lam = lam
    
  def sigmoid(self, x):
    '''
    The sigmoid function.

    Parameters
    ----------
    x : float
      Real number input.

    Returns
    -------
    float
      Value of sigmoid at x.

    '''
    return 1.0/(1+ np.exp(-x))

  def sigmoid_derivative(self, x):
    '''
    Derivative of sigmoid function.

    Parameters
    ----------
    x : float
      Real number input.

    Returns
    -------
    float
      Value of gradient of sigmoid at x.

    '''
    return x * (1.0 - x)
    
  def flatten(self, arrays):
    '''
    Flattens weight matrices into a vector to be supplied to optimiser.

    Parameters
    ----------
    arrays : list
      List of numpy arrays, each corresponding to a layer of the network.

    Returns
    -------
    flat : np.array
      Vector containing the values from the list.

    '''
    flat = np.array([])
    for arr in arrays:
      flat = np.append(flat, np.reshape(arr, (-1,)))
    return flat
  
  def un_flatten(self, flat, shapes):
    '''
    Reverses flatten, converts output from optimiser to updates for neural net.

    Parameters
    ----------
    flat : np.array
      Vector to be reshaped into list of arrays.
    shapes : list
      List of shapes that the arrays should be.

    Returns
    -------
    arrs : list
      List of array containing values from the input.

    '''
    arrs = []
    i=0
    for n, m in shapes:
      arrs.append(np.array(flat[i:i+n*m]).reshape(n,m))
      i += n*m
    return arrs

  def feedforward(self, weights):
    '''
    Propagates the input through the network.

    Parameters
    ----------
    weights : list
      List of weight matrices.

    Returns
    -------
    None.

    '''
    self.weights = weights
    self.layer_values = [self.input] #Get first layer
    for layer in range(0, self.n_hidden+1):
      self.layer_values += [self.sigmoid(np.dot(self.layer_values[-1], self.weights[layer]))] #Iteratively append outputs for each layer
    self.output = self.layer_values[-1]
    
  def loss(self, weights):
    '''
    We use the square loss, could experiment with others, but we will do regression so this seems sensible.

    Parameters
    ----------
    weights : list
      List of weight matrices.

    Returns
    -------
    float
      Square loss for these weights.

    '''
    self.feedforward(weights)
    output_error = self.y - self.output
    return np.mean(output_error**2) + 0.5 * self.lam * np.sum([np.sum(weight**2) for weight in weights])

  def gradients(self, weights):
    '''
    Get the gradient of the loss to be supplied to optimiser.

    Parameters
    ----------
    weights : list
      List of weight matrices.

    Returns
    -------
    adjustments : list
      List of gradients.

    '''
    self.feedforward(weights)
    ### First get error, delta and adjustment for last layer
    output_error = self.y - self.layer_values[self.n_hidden + 1]
    output_delta = output_error * self.sigmoid_derivative(self.layer_values[self.n_hidden + 1])
    output_adjustment = - np.transpose(self.layer_values[self.n_hidden]). dot(output_delta)
    errors = [output_error]
    deltas = [output_delta]
    adjustments = [output_adjustment]
    ### Then go back through the layes, calculating the adjustments needed for each set of weights
    for layer in reversed(range(1, self.n_hidden+1)):
      layer_error = deltas[0].dot(np.transpose(self.weights[layer]))
      layer_delta = layer_error * self.sigmoid_derivative(self.layer_values[layer])
      layer_adjustment = - (1/np.shape(self.y)[0])*np.transpose(self.layer_values[layer-1]).dot(layer_delta)
      errors = [layer_error] + errors
      deltas = [layer_delta] + deltas
      adjustments = [layer_adjustment] + adjustments
      #print([np.mean(grad) for grad in adjustments])
    adjustments = [adjustments[i] + self.lam * weights[i] for i in range(len(weights))]
    return adjustments

  def predict(self, x):
    '''
    Predicts the labels of new data.

    Parameters
    ----------
    x : np.array
      New data.

    Returns
    -------
    None.

    '''
    self.layer_pred = x
    for weight_matrix in self.weights:
      self.layer_pred = self.sigmoid(np.dot(self.layer_pred, weight_matrix))
    self.output_pred = self.layer_pred

