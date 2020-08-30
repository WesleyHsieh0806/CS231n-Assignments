from builtins import range
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
    dW = np.zeros(W.shape)
    num_sample = X.shape[0]
    num_class = W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for i in range(num_sample):

        raw_score = X[i].dot(W)
        # shift the value by np.max_score to prevent value instability ->
        # This may cause the gradient raw_score_max to be different than how we compute
        # For the sake of convenience, we will just ignore it here.
        raw_score -= np.max(raw_score)
        exp_score = np.exp(raw_score)
        # compute the sum of exponential value as discriminator
        denominator = np.sum(exp_score)
        score = exp_score / np.maximum(denominator, 1e-8)
        for j in range(num_class):
            if j != y[i]:
                # dW[k,j] = x[i,k] * dloss/dscore[true_class](-1/score) * dscore[true_class]/draw_score[j](-score[true]*(score[j]))
                # after simplifying, we can obtain w[k,j] = X[i,k] * score[j]
                dW[:, j] += X[i, :] * \
                    (score[j])
            else:
                # after simplifying, we can obtain w[k,j] = X[i,k] * (score[j]-1)
                dW[:, j] += X[i, :] * \
                    (score[y[i]]-1)
        loss -= (np.log(score[y[i]]))
    # divide the loss by num_sample
    loss /= num_sample
    dW /= num_sample
    # regularization term
    loss += reg*np.sum(W**2)
    dW += reg*2*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros(W.shape)
    # N: number of samples
    N = X.shape[0]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    ''' dW = X.T @ draw_scores'''
    ''' Forward'''
    # compute the output(raw_score) of linear layer ->(N, C)
    raw_score = X.dot(W)
    # compute the output of softmax layers
    exp_score = np.exp(raw_score)
    scores = exp_score / np.expand_dims(np.sum(exp_score, axis=1), 1)
    # compute the loss by score
    loss = np.sum(-np.log(scores[np.arange(N), y]))
    loss /= N
    loss += reg*np.sum(W**2)
    '''Backward pass '''
    # After derivation, we can learn that draw_score[i][j] will be scores[i][j] (j!=y[i])
    # and scores[i][j]-1 (j=y[i])
    draw_scores = scores.copy()
    draw_scores[np.arange(N), y] -= 1
    # dW = X.T@ draw_scores/N + 2*W*reg
    dW = X.T @ draw_scores
    dW /= N
    dW += 2*W*reg
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
