import numpy as np
from random import shuffle

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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  num_dim =W.shape[0]
  p = np.zeros((1,num_classes)) #probability
  for i in xrange(num_train):
     scores = X[i].dot(W)
     scores -= np.max(scores)
     p = np.exp(scores)/np.sum(np.exp(scores))
     loss += -np.log(p[y[i]])
     p[y[i]] -= 1
     dW += np.dot(np.reshape(X[i],(num_dim,1)),np.reshape(p,(1,num_classes)))
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW = (1.0/num_train)*dW + reg*W     
     
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_dim =W.shape[0]

  scores = X.dot(W)
  scores -= np.max(scores,axis = 1).reshape(num_train,1) 
  p = np.exp(scores)/(np.sum(np.exp(scores),axis = 1).reshape(num_train,1))
  loss += -np.sum(np.log(p[(range(num_train),y)]))
  loss /= num_train 
  loss += 0.5 * reg * np.sum(W * W)
  
  p[(range(num_train),y)] -=1
  dW = (1.0/num_train)*np.dot(X.T,p) + reg*W    
     
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

