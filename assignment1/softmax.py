from cs231n.gradient_check import grad_check_sparse
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from cs231n.classifiers.softmax import softmax_loss_naive
from cs231n.classifiers.softmax import softmax_loss_vectorized
from cs231n.gradient_check import eval_numerical_gradient
from cs231n.data_utils import load_CIFAR10
''' Load the data '''
cifar_path = os.path.join(os.path.dirname(
    __file__), 'cs231n', 'datasets', 'cifar-10-batches-py')
X_train, Y_train, X_test, Y_test = load_CIFAR10(cifar_path)
# check the size
print("Size of training data:{}".format(X_train.shape))
print("Size of training label:{}".format(Y_train.shape))
print("Size of testing data:{}".format(X_test.shape))
print("Size of testing label:{}".format(Y_test.shape))

''' split the data into train, val and test set'''
num_train = 49000
num_val = 1000
num_test = 1000
# we can also select development set from train data
num_dev = 1000

# randomly select num_val data from train dataset
val_mask = np.zeros(len(X_train), dtype=bool)
val_mask[np.random.choice(
    range(len(X_train)), num_val, replace=False)] = True
val_x = X_train[val_mask]
val_y = Y_train[val_mask]

# the remaining train data are train data
train_x = X_train[~val_mask]
train_y = Y_train[~val_mask]
# randomly select num_test data from test dataset
test_mask = np.random.choice(range(len(X_test)), num_test, replace=False)
test_x = X_test[test_mask]
test_y = Y_test[test_mask]

''' Preprocess the data (flattening)'''
train_x = train_x.reshape(train_x.shape[0], -1)
val_x = val_x.reshape(val_x.shape[0], -1)
test_x = test_x.reshape(test_x.shape[0], -1)

# Subtract the mean from data
train_x -= np.mean(train_x, axis=0)
val_x -= np.mean(train_x, axis=0)
test_x -= np.mean(train_x, axis=0)

# Append the bias term into data so that SVM classifier only needs to worry about W
train_x = np.hstack([train_x, np.ones([train_x.shape[0], 1])])
val_x = np.hstack([val_x, np.ones([val_x.shape[0], 1])])
test_x = np.hstack([test_x, np.ones([test_x.shape[0], 1])])
# randomly select num_dev data from train data
dev_mask = np.random.choice(range(len(train_x)), num_dev, replace=False)
dev_x = train_x[dev_mask]
dev_y = train_y[dev_mask]

print("Train data shape:{}".format(train_x.shape))
print("Train label shape:{}".format(train_y.shape))
print("Val data shape:{}".format(val_x.shape))
print("Val label shape:{}".format(val_y.shape))
print("Test Data shape:{}".format(test_x.shape))
print("Test Label shape:{}".format(test_y.shape))

''' 
* Implement softmax_loss_naive and softmax_loss_vectorized
'''
''' softmax_loss_naive '''
# Randomly initialize the W
num_class = np.max(train_y)+1
W = np.random.randn(dev_x.shape[1], num_class) / np.sqrt(dev_x.shape[1])
loss, grad = softmax_loss_naive(W, dev_x, dev_y, reg=0.)
print("loss:{:.4f} -log(0.1):{:.4f}".format(loss, -np.log(0.1)))


def f(w):
    return softmax_loss_naive(w, dev_x, dev_y, reg=0.)[0]


# Check the correctness of gradient computation
# this function will compute the numerical gradient and compare it to analytical gradient
grad_check_sparse(f, W, analytic_grad=grad, num_checks=5)

''' softmax_loss_vectorized'''
loss_vec, grad_vec = softmax_loss_vectorized(W, dev_x, dev_y, reg=0.)

# Check the correctness of gradient computation
# this function will compute the numerical gradient and compare it to analytical gradient
grad_check_sparse(f, W, analytic_grad=grad_vec, num_checks=5)

print("{:=^100}".format("Gradient check complete!"))
print("{:=^100}".format("Compare the softmax_naive with softmax_vectorized"))
print("Loss difference:{:8f}".format(loss-loss_vec))
print("Norm of Gradient difference:{:8f}".format(
    np.linalg.norm(grad_vec-grad)))
