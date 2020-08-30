import numpy as np
import matplotlib.pyplot as plt
import os
import time
from cs231n.data_utils import load_CIFAR10
from cs231n.classifiers.linear_svm import svm_loss_naive
from cs231n.classifiers.linear_svm import svm_loss_vectorized
from cs231n.gradient_check import grad_check_sparse


def partition_kcross_validation(x, y, k):
    ''' 
    * partition the data into a list which contains k folds of data
    * x: data
    * y: label of data
    * Return: Two lists :1. data folds 2.label folds ;each containing k folds of data
    '''
    size_of_fold = len(x)//k
    data_folds = [x[i*size_of_fold:(i+1)*size_of_fold] for i in range(k-1)]
    data_folds.append(x[(k-1)*size_of_fold:])
    label_folds = [y[i*size_of_fold:(i+1)*size_of_fold] for i in range(k-1)]
    label_folds.append(y[(k-1)*size_of_fold:])

    return data_folds, label_folds


def main():
    ''' Load the data '''
    cifar_path = os.path.join(os.path.dirname(
        __file__), 'cs231n', 'datasets', 'cifar-10-batches-py')
    X_train, Y_train, X_test, Y_test = load_CIFAR10(cifar_path)
    # check the size
    print("Size of training data:{}".format(X_train.shape))
    print("Size of training label:{}".format(Y_train.shape))
    print("Size of testing data:{}".format(X_test.shape))
    print("Size of testing label:{}".format(Y_test.shape))

    ''' visualize some image '''
    # classes = ['plane', 'car', 'bird', 'cat', 'deer',
    #            'dog', 'frog', 'horse', 'ship', 'truck']
    # num_classes = len(classes)
    # sample_per_class = 3

    # for class_number, class_name in enumerate(classes):
    #     # class_index: the index of samples which belongs to class_name
    #     class_index = np.flatnonzero(train_y == class_number)
    #     sample_indices = np.random.choice(
    #         class_index, sample_per_class, replace=False)
    #     for sample_order in range(sample_per_class):
    #         # Plot the images of each class(Total:|sample_per_class| images)
    #         plt.subplot(sample_per_class, len(classes), len(
    #             classes)*sample_order + class_number + 1)
    #         plt.imshow(train_x[sample_indices[sample_order]].astype('uint8'))
    #         plt.axis('off')
    # plt.show()
    # plt.close()

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
    val_x -= np.mean(val_x, axis=0)
    test_x -= np.mean(test_x, axis=0)

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
    * SVM Classifier 
    '''
    # Initialize W and reg
    W = np.random.randn(train_x.shape[1], 10)
    reg = 5

    ''' check the correctness of grad'''
    # f is a function which computes the loss. We can compute the numerical gradient with it.
    # def f(w): return svm_loss_naive(w, dev_x, dev_y, reg=reg)[0]
    # grad_check_sparse(f, W, grad)
    # def f(w): return svm_loss_vectorized(w, dev_x, dev_y, reg=reg)[0]
    # grad_check_sparse(f, W, grad)
    ''' Compare the execution time between non-vectorized and vectorized loss function'''
    start = time.time()
    # SVM_Loss_naive(W, trainx, trainy, regularize term strength)
    loss, grad = svm_loss_naive(W, dev_x, dev_y, reg=reg)
    time_naive = time.time() - start
    print("Loss from svm_loss_naive:{:.4f} time:{:.8f}".format(
        loss, time_naive))
    start = time.time()
    loss2, grad2 = svm_loss_vectorized(W, dev_x, dev_y, reg=reg)
    time_vectorized = time.time()-start
    print("Loss from svm_loss_vectorized:{:.4f} time:{:.8f}".format(
        loss, time_naive))

    ''' Train the SVM Classifier '''
    from cs231n.classifiers import LinearClassifier
    svm = LinearClassifier()
    loss_list = svm.train(train_x, train_y, learning_rate=1e-7,
                          reg=2.5e4, num_iters=1500, verbose=True)
    # print the loss of the training process
    plt.plot(loss_list, color='b', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title("Loss")
    # plt.show()
    plt.close()

    ''' Predict the results with the trained SVM model'''
    val_pred = svm.predict(val_x)
    print("The accuracy on validation set:{:.4f}".format(
        np.mean(val_y == val_pred)))

    ''' Use k-Cross-Validation to fine-tune the hyperparameter '''
    K = 4
    average_time = 2
    data_folds, label_folds = partition_kcross_validation(train_x, train_y, K)
    # Record the result for each parameter
    # results[(learning rate, reg)] = (val accuracy)
    lrs = np.random.uniform(1e-7, 1e-5, size=3)
    regs = np.random.uniform(1e3, 1e5, size=3)
    results = {}
    for Time in range(average_time):
        # select the first fold as validation data
        val_x = data_folds[0]
        val_y = label_folds[0]
        train_x = np.concatenate(data_folds[1:], axis=0)
        train_y = np.concatenate(label_folds[1:], axis=0)
        for lr in lrs:
            for reg in regs:
                # train the model with each parameter setup
                svm = LinearClassifier()
                loss_list = svm.train(train_x, train_y, learning_rate=lr,
                                      reg=reg, num_iters=100, verbose=False)
                val_y_pred = svm.predict(val_x)
                val_acc = np.mean(val_y_pred == val_y)
                # Record the accuracy in the dictionary
                if (lr, reg) not in results:
                    results[(lr, reg)] = val_acc/average_time
                else:
                    results[(lr, reg)] += val_acc/average_time
                print("Learning rate:{} Regularization:{} Acc:{:.4f}".format(
                    lr, reg, val_acc))
        # move the first fold to the last position in the list
        data_folds.append(data_folds[0])
        data_folds = data_folds[1:]
        label_folds.append(label_folds[0])
        label_folds = label_folds[1:]
    # plot the accuracy of each parameter and output the best parameter
    best_acc = -1.
    best_lr = -1
    best_reg = -2
    print("Take {} times results of average".format(average_time))
    for (lr, reg), val_acc in results.items():
        print("Learning rate:{} Regularization:{} Acc:{:.4f}".format(
            lr, reg, val_acc))
        if val_acc > best_acc:
            best_acc = val_acc
            best_lr = lr
            best_reg = reg
    print("Best lr:{} Best Reg:{}".format(best_lr, best_reg))

    ''' Visualize the result of cross-validation and parameter setting'''
    import math
    # xaxis: lr in log scale
    # yaxis: reg in log scale
    x_scatter = [math.log10(x[0]) for x in results]
    y_scatter = [math.log10(x[1]) for x in results]

    marker_size = 100
    # Transfer the accuracy to color so that we can visualize the accuracy with color
    colors = [results[x] for x in results]
    plt.scatter(x_scatter, y_scatter, s=marker_size,
                c=colors, cmap=plt.cm.coolwarm)
    plt.colorbar()
    plt.xlabel("log learning rate")
    plt.ylabel("log regularization strength")
    plt.title('Validation accuracy')
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
