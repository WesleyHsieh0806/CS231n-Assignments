import numpy as np
import matplotlib.pyplot as plt
import time
import os
from cs231n.classifiers.neural_net import TwoLayerNet
from cs231n.gradient_check import eval_numerical_gradient
from cs231n.data_utils import load_CIFAR10


def relative_error(grad_x, grad_y):
    ''' Return relative error to evaluate the correctness of gradient'''
    return np.mean(np.abs(grad_x-grad_y)/np.maximum(1e-8, np.abs(grad_x) + np.abs(grad_y)))


''' Load the data '''
cifar_path = os.path.join(os.path.dirname(
    __file__), 'cs231n', 'datasets', 'cifar-10-batches-py')
train_x, train_y, test_x, test_y = load_CIFAR10(cifar_path)
# Flatten the data
train_x = train_x.reshape(len(train_x), -1)
test_x = test_x.reshape(len(test_x), -1)
# check the size
print("Size of training data:{}".format(train_x.shape))
print("Size of training label:{}".format(train_y.shape))
print("Size of testing data:{}".format(test_x.shape))
print("Size of testing label:{}".format(test_y.shape))

''' Parameter setup'''
input_size = train_x.shape[1]
hidden_size = 2
output_size = np.max(train_y)+1

'''Gradient check'''
# Subtract the mean from data
train_x = train_x - np.mean(train_x, axis=0)
test_x = test_x - np.mean(train_x, axis=0)

# Subsample the dev data from train data
dev_size = 1000
dev_index = np.random.choice(np.arange(len(train_x)), dev_size, replace=False)
dev_x = train_x[dev_index]
dev_y = train_y[dev_index]

# check the gradient
model = TwoLayerNet(input_size, hidden_size, output_size)
loss, grad = model.loss(dev_x, dev_y, reg=0.0)


def f(w): return model.loss(dev_x, dev_y, reg=0.0)[0]

''' Compute the numerical gradient for each parameter one by one'''
# for para_name in model.params:
#     '''
#     * eval_numerical_gradient(f, x) slightly changes x to compute dL/dx
#     * f computes the loss we defined
#     '''
#     # Compute the numerical gradient
#     # Notice that x in this function is the weights/bias
#     # After sending x, this function will iterate through all the elements in x to compute the gradients
#     grad_numerical = eval_numerical_gradient(
#         f, model.params[para_name])
#     print("Parameter Name:{} |Relatvie error:{:.4f}".format(
#         para_name, relative_error(grad_numerical, grad[para_name])))
''' Train a neural network'''
input_size = train_x.shape[1]
hidden_size = 500
output_size = np.max(train_y)+1

final_model = TwoLayerNet(input_size, hidden_size, output_size)

# Subsample the dev data from train data
dev_size = 10000
dev_index = np.random.choice(np.arange(len(train_x)), dev_size, replace=False)
dev_x = train_x[dev_index]
dev_y = train_y[dev_index]
# Subsample the test data
sub_test_size = 5000
sub_test_index = np.random.choice(
    range(len(test_x)), sub_test_size, replace=False)
sub_test_x = test_x[sub_test_index]
sub_test_y = test_y[sub_test_index]


# Train the network
history = final_model.train(dev_x, dev_y, sub_test_x, sub_test_y, learning_rate=1e-3,
                            learning_rate_decay=0.95, num_iters=1500, batch_size=32, verbose=True)
# the returned dictionary contains the history of loss and accuracy
loss = history['loss_history']
train_acc = history['train_acc_history']
test_acc = history['val_acc_history']

# plot the loss history
plt.plot(loss, color='b', marker='o', linestyle='-', label='loss')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('Cross-Entropy')
plt.show()
plt.close()
# plot the accuracy history
plt.plot(train_acc, color='b', marker='o', linestyle='-', label='train acc')
plt.plot(test_acc, color='b', marker='o', linestyle='-', label='test acc')
plt.xlabel('iteration')
plt.ylabel('acc')
plt.title('Accuracy')
plt.show()
plt.close()
