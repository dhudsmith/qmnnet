#!/usr/bin/env python

"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.

This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html

More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""

from __future__ import print_function

import sys
import time

import numpy as np
import theano
import theano.tensor as T

from sklearn.model_selection import train_test_split

import lasagne


# ################## Load the potentials and eigenvalues  ##################

def load_dataset(filepath, num_target, test_frac = 0.25, val_frac = 0.25, random_state = np.random.randint(0,1000)):

    # Read in the file at `filepath'
    data = np.load(filepath)
    X = data[::, :-num_target]
    y = data[::, -num_target:]

    # Split train and test
    X_train, X_testval, y_train, y_testval = train_test_split(
        X, y,
        test_size=test_frac + val_frac,
        random_state=random_state
    )

    X_test, X_val, y_test, y_val = train_test_split(
        X_testval, y_testval,
        test_size= test_frac / (test_frac + val_frac),
        random_state= random_state
    )

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test


# ##################### Build the neural network model #######################

def build_mlp(input_shape, num_outputs, hidden_layer_sizes, drop_input=.2,
              drop_hidden=.5,
              input_var = None):
    # Note that to make the code easier,
    # all the layers are just called `network` -- there is no need to give them
    # different names if all we return is the last one we created anyway; we
    # just used different names above for clarity.

    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
    network = lasagne.layers.InputLayer(shape=input_shape,
                                        input_var=input_var)
    if drop_input:
        network = lasagne.layers.dropout(network, p=drop_input)

    # Hidden layers and dropout:
    nonlin = lasagne.nonlinearities.rectify
    for size in hidden_layer_sizes:
        network = lasagne.layers.DenseLayer(
            network, size, nonlinearity=nonlin)
        if drop_hidden:
            network = lasagne.layers.dropout(network, p=drop_hidden)

    # Output layer:
    network = lasagne.layers.DenseLayer(network,
                                        num_units=num_outputs,
                                        nonlinearity=None)
    return network



# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# Notice that this function returns only mini-batches of size `batchsize`.
# If the size of the data is not a multiple of `batchsize`, it will not
# return the last (remaining) mini-batch.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(filepath, num_target, hidden_layer_sizes, num_epochs=500):
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = \
        load_dataset(filepath=filepath,
                     num_target=num_target)

    # Prepare Theano variables for inputs and targets
    input_var = T.matrix('inputs')
    target_var = T.matrix('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")

    network = build_mlp(input_shape = (None, X_train.shape[1]),
                        num_outputs = y_train.shape[1],
                        hidden_layer_sizes = hidden_layer_sizes,
                        input_var = input_var)


    # Create a loss expression for training.
    # We use MSE for this regression problem.
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction,
                                                 target_var)
    test_loss = test_loss.mean()

    test_acc = lasagne.objectives.squared_error(test_prediction, target_var)
    test_acc = test_acc.mean()

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Create an expression to output the error per eigenvalue
    test_err_list = lasagne.objectives.squared_error(test_prediction, target_var)
    test_err_list = test_err_list.mean(axis = 0)
    eig_err_fn = theano.function([input_var, target_var], test_err_list)

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_eigs_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_eigs_err += eig_err_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_eigs_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_eigs_err += eig_err_fn(inputs, targets)
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  training err. per eig.:\n", train_eigs_err / train_batches)
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation err. per eig.:\n", val_eigs_err / val_batches)

    # After training, we compute and print the test error:
    test_err = 0
    test_eigs_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_eigs_err += eig_err_fn(inputs, targets)
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test err. per eig.:\n", test_eigs_err / test_batches)

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    # if ('--help' in sys.argv) or ('-h' in sys.argv):
    #     print("Trains a neural network on MNIST using Lasagne.")
    #     print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
    #     print()
    #     print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
    #     print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
    #     print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
    #     print("       input dropout and DROP_HID hidden dropout,")
    #     print("       'cnn' for a simple Convolutional Neural Network (CNN).")
    #     print("EPOCHS: number of training epochs to perform (default: 500)")
    # else:
    #     kwargs = {}
    #     if len(sys.argv) > 1:
    #         kwargs['model'] = sys.argv[1]
    #     if len(sys.argv) > 2:
    #         kwargs['num_epochs'] = int(sys.argv[2])
    #     main(**kwargs)
    filepath = 'Data/potentialGrid_100000_NB10_lam0.75_V2010.npy'
    num_target = 10
    hidden_layer_sizes = (200,)
    num_epochs = 50

    main(filepath, num_target, hidden_layer_sizes, num_epochs)

