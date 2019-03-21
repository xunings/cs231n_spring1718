import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  n_train = X.shape[0] #N
  n_pixel = X.shape[1] #D
  n_class = W.shape[1] #C
  
  for i_train in range(n_train):
    scores = X[i_train,:]@W
    exp_scores = np.exp(scores)
    probs = exp_scores/sum(exp_scores)
    loss += -np.log( probs[y[i_train]] )

    # loss = -log(e^x_correct/sum_i(e^x_i))=log(sum_i(e^x_i))-x_correct, x is the score.
    dScores =  exp_scores/sum(exp_scores) #first term
    dScores[y[i_train]] += -1 #second term for x_correct
    # Back propagation through dScore
    dW += X[i_train,:].reshape(n_pixel,1) @ dScores.reshape(1, n_class) 
    assert(dW.shape == (n_pixel, n_class))
    
  
  loss /= n_train
  dW /= n_train
  loss += reg * np.sum(np.square(W))
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  n_train = X.shape[0] #N
  n_class = W.shape[1] #C
  

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X@W
  exp_scores = np.exp(scores)
  probs = exp_scores/np.sum(exp_scores, axis=1).reshape(n_train,1)
  probs_correct = probs[list(range(n_train)),y]
  loss += -np.sum( np.log(probs_correct) )
  loss /= n_train
  loss += reg * np.sum(np.square(W))

  # As seen in the non-vectorized version, the first term of dScore happens
  # to be the same as the probs.
  dScores = probs 
  dScores[list(range(n_train)),y] += -1
  assert(dScores.shape==(n_train, n_class))
  # DxN@NxC=DxC. It is simply the sum of (X@dScore) across multiple training example,
  # since DxN can be divided to N columns, and NxC can be divided to N rows,
  # and the sum can be understood as a result of the block matrix mulitplication.
  dW = X.T@dScores / n_train
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

