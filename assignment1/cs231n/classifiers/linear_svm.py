from builtins import range
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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                # dw:(Din, num_classes)
                # loss = max(Sj - Strue +1, 0)
                # This case only considers loss caused by score j of sample i;
                # as a result, we only need to compute the gradient causes by Sj and Sy[i]
                # dL/dSj = 1 dSj/dWk,j = sigma(xi,k) (1<=i<=num_sample, 1<=k<=D)
                dW[:, j] += X[i, :]
                dW[:, y[i]] -= X[i, :]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # the gradient from regularization term
    dW += reg*2*W
    # the gradient from the loss function should be computed with the loss above,
    # so we just modify the code above
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    Scores = X.dot(W)
    # correct_score:(num_sample, )
    correct_scores = Scores[np.arange(len(y)), y]
    # all_loss : same shape as Scores
    all_loss = np.maximum(0, Scores - correct_scores.reshape(-1, 1) + 1)
    # cancel the loss from true class
    # all_loss:(num_sample, num_classes) all_loss[i][j] indicates loss caused by scorei,j
    all_loss[np.arange(len(y)), y] = 0
    # Sum up the loss and divide it by num_sample
    loss += np.sum(all_loss)
    loss /= X.shape[0]

    # Add the Regularization term to the loss
    loss += reg*np.sum(W**2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW += 2*reg*W
    ''' Use Matrix Caculus to compute dW '''
    # first, compute dScores
    dScores = np.ones(Scores.shape)
    dScores[all_loss > 0] = 1
    # notice that dScores_trueclass should be accumulated
    dScores[np.arange(len(y)), y] -= np.sum((all_loss > 0), axis=1)

    # Second, compute dW by matrix caculus
    dW = X.T@dScores
    # divide dW by num_sample
    dW /= X.shape[0]
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
