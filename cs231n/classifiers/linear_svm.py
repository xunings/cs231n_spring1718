import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1] # C
  num_train = X.shape[0] # N
  num_pixels = W.shape[0] # D
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:        
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        # loss = score_j - score_correct + 1
        # for each score, the gradient is simply X[i].
        loss += margin
        dW[:,j] += X[i]
        dW[:,y[i]] -= X[i] 

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2*reg*W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  num_classes = W.shape[1] # C
  num_train = X.shape[0] # N
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X@W # NxD @ DxC = NxC
  # Ref: https://stackoverflow.com/questions/17074422/select-one-element-in-each-row-of-a-numpy-array-by-column-indices
  scores_correct = np.choose(y, scores.T)
  scores_SVM = scores - scores_correct.reshape(num_train,1)
  scores_SVM = scores_SVM + 1.
  #scores_SVM = scores_SVM[scores_SVM>0] #this one returns a 1-D array
  scores_SVM[scores_SVM<0]=0
  scores_SVM[list(range(num_train)),y] = 0
   
  # -1 for the correct ones.
  loss = np.sum(scores_SVM) / num_train
  loss += reg * np.sum(W * W)
  
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  score_entries_grad = np.ones((num_train, num_classes))*(scores_SVM>0)
  num_entries_above0 = np.sum(score_entries_grad, axis=1)
  score_entries_grad[list(range(num_train)),y]=-num_entries_above0
  dW = X.T@score_entries_grad
  dW /= num_train
  dW += 2*reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
