import numpy as np
import matplotlib.pyplot as plt
from cs231n.data_utils import load_CIFAR10
import os
import time


def partition_crossvalidation(train_x, train_y, number_of_folds):
    '''
    Input: train_x, train_y: training data
    Return: A list which contains number_of_folds folds of data
    '''
    size = len(train_x)//number_of_folds
    train_x_fold = [
        train_x[i*size:(i+1) * size] for i in range(number_of_folds-1)]
    train_x_fold.append(train_x[(number_of_folds-1)*size:])
    train_y_fold = [
        train_y[i*size:(i+1) * size] for i in range(number_of_folds-1)]
    train_y_fold.append(train_y[(number_of_folds-1)*size:])

    return train_x_fold, train_y_fold


def main():
    ''' Load the data '''
    cifar_path = os.path.join(os.path.dirname(
        __file__), 'cs231n', 'datasets', 'cifar-10-batches-py')
    train_x, train_y, test_x, test_y = load_CIFAR10(cifar_path)
    # check the size
    print("Size of training data:{}".format(train_x.shape))
    print("Size of training label:{}".format(train_y.shape))
    print("Size of testing data:{}".format(test_x.shape))
    print("Size of testing label:{}".format(test_y.shape))

    ''' Visualize some images'''
    classes = ['plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    sample_per_class = 3

    for class_number, class_name in enumerate(classes):
        # random select |sample_per_class| images which belongs to "class_name"
        # class_index: the index of images which belongs to this class
        class_index = np.flatnonzero(train_y == class_number)
        idx = np.random.choice(class_index, sample_per_class, replace=False)
        for sample in range(sample_per_class):
            # sample:0 -> the first sample of this class
            image = train_x[idx[sample]].astype('uint8')
            plt.subplot(sample_per_class, num_classes,
                        class_number + sample*num_classes + 1)
            plt.imshow(image)
            # removes the axis and edges
            plt.axis("off")
            if sample == 1:
                plt.title(class_name)
    # plt.show()
    plt.close()

    ''' Subsample some data(optional) and flatten the data'''
    # subsample size
    number_of_train = 500
    number_of_test = 100
    # random select the index
    train_index = np.random.choice(
        np.arange(len(train_x)), number_of_train, replace=False)
    test_index = np.random.choice(
        np.arange(len(test_x)), number_of_test, replace=False)
    # Save the total data in other variable for future use
    train_x_all = train_x.reshape(train_x.shape[0], -1)
    train_y_all = train_y.reshape(train_y.shape[0])
    test_x_all = test_x.reshape(test_x.shape[0], -1)
    test_y_all = test_y.reshape(test_y.shape[0])
    # subsample
    train_x = train_x[train_index]
    train_y = train_y[train_index]
    test_x = test_x[test_index]
    test_y = test_y[test_index]

    train_x = train_x.reshape(train_x.shape[0], -1)
    train_y = train_y.reshape(train_y.shape[0])
    test_x = test_x.reshape(test_x.shape[0], -1)
    test_y = test_y.reshape(test_y.shape[0])

    ''' train the knn classifier'''
    from cs231n.classifiers import KNearestNeighbor
    classifier = KNearestNeighbor()
    classifier.train(train_x, train_y)

    ''' test the implementation '''
    # Function: compute_distances_two_loops
    start = time.time()
    dists_2 = classifier.compute_distances_two_loops(test_x)
    print(dists_2.shape)
    print("Execution time of compute_distances_two_loops:{}".format(time.time()-start))

    # Function:compute_distances_one_loop
    start = time.time()
    dists_1 = classifier.compute_distances_one_loop(test_x)
    print(dists_1.shape)
    print("Execution time of compute_distances_one_loop:{}".format(time.time()-start))

    # Function:compute_distances_no_loop
    start = time.time()
    dists = classifier.compute_distances_no_loops(test_x)
    print(dists.shape)
    print("Execution time of compute_distances_no_loop:{}".format(time.time()-start))

    # visualize distance matrix
    plt.imshow(dists.astype('uint8'))
    # plt.show()
    plt.close()

    ''' Predict the data '''
    for K in range(1, 10):
        y_pred = classifier.predict(test_x, k=K, num_loops=0)
        acc = np.mean(test_y == y_pred)
        print("K={} Accuracy:{:.4f}".format(K, acc))

    ''' Check the correctness of distance matrix'''
    difference = np.linalg.norm(dists-dists_1)
    if difference < 0.001:
        print("Two difference matrix are the same!")
    else:
        print("Funcition bug!")

    num_folds = 5
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

    X_train_folds = []
    y_train_folds = []
    ################################################################################
    # TODO:                                                                        #
    # Split up the training data into folds. After splitting, X_train_folds and    #
    # y_train_folds should each be lists of length num_folds, where                #
    # y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
    # Hint: Look up the numpy array_split function.                                #
    ################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    X_train_folds, y_train_folds = partition_crossvalidation(
        train_x, train_y, num_folds)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # A dictionary holding the accuracies for different values of k that we find
    # when running cross-validation. After running cross-validation,
    # k_to_accuracies[k] should be a list of length num_folds giving the different
    # accuracy values that we found when using that value of k.
    k_to_accuracies = {}

    ################################################################################
    # TODO:                                                                        #
    # Perform k-fold cross validation to find the best value of k. For each        #
    # possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
    # where in each case you use all but one of the folds as training data and the #
    # last fold as a validation set. Store the accuracies for all fold and all     #
    # values of k in the k_to_accuracies dictionary.                               #
    ################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for val_index in range(num_folds):
        # val_index: the index of data fold in train_x_fold
        val_x_cross = X_train_folds[val_index]
        val_y_cross = y_train_folds[val_index]
        # Since the training data for cross validation should not overlap with val data
        # We have to deal with some marginal cases
        if val_index == 0:
            train_x_cross = np.concatenate(X_train_folds[1:], axis=0)
            train_y_cross = np.concatenate(y_train_folds[1:], axis=0)
        elif val_index == (num_folds-1):
            train_x_cross = np.concatenate(X_train_folds[:num_folds-1], axis=0)
            train_y_cross = np.concatenate(y_train_folds[:num_folds-1], axis=0)
        else:
            # folds = [train data| val data | train data]
            array1 = np.concatenate(X_train_folds[:val_index], axis=0)
            array2 = np.concatenate((X_train_folds[val_index+1:]), axis=0)
            train_x_cross = np.concatenate([array1, array2], axis=0)
            array1 = np.concatenate(y_train_folds[:val_index], axis=0)
            array2 = np.concatenate((y_train_folds[val_index+1:]), axis=0)
            train_y_cross = np.concatenate([array1, array2], axis=0)
        print("Size of training data:{}".format(train_x_cross.shape[0]))
        print("Size of val data:{}".format(val_x_cross.shape[0]))

        # test the accuracy for different parameter
        for k in k_choices:
            if k not in k_to_accuracies:
                k_to_accuracies[k] = []
            model = KNearestNeighbor()
            model.train(train_x_cross, train_y_cross)
            # predict the val data with different parameters
            val_pred = model.predict(val_x_cross, k=k)
            acc = np.mean(val_pred == val_y_cross)
            # Insert the accuracy into lists
            k_to_accuracies[k].append(acc)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Print out the computed accuracies
    for k in sorted(k_to_accuracies):
        for accuracy in k_to_accuracies[k]:
            print('k = %d, accuracy = %f' % (k, accuracy))

    ''' Plot the Accuracy and compute the mean'''
    accuracy_mean = []
    accuracy_std = []
    for k in sorted(k_to_accuracies):
        # initial k_scatter to plot the scatter for each value in k_choices
        k_scatter = [k for i in range(len(k_to_accuracies[k]))]
        accuracies = k_to_accuracies[k]
        plt.scatter(k_scatter, accuracies)
        # compute the mean and std of accuracy for each choice of k
        accuracy_mean.append(np.mean(accuracies))
        accuracy_std.append(np.std(accuracies))
    # error bar can help us describe the meaning and standard deviation of accuracy
    # for example, by providing standard deviation, the error bar shows us mean +/- standard
    plt.errorbar(k_choices, accuracy_mean, yerr=accuracy_std)
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.title("KNN Cross-Validation")
    # plt.show()
    plt.close()

    ''' Train the best model'''
    best_k = 8
    print("{:=^40}".format("Train the model with the best observed k"))
    best_model = KNearestNeighbor()
    best_model.train(train_x_all, train_y_all)
    best_y_pred = best_model.predict(test_x_all, k=best_k)
    acc = np.mean(best_y_pred == test_y_all)
    print("Best Model with k={}:{:.4f}".format(best_k, acc))


if __name__ == "__main__":
    main()
