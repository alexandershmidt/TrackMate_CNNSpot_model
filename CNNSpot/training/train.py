import shutil
import numpy as np
import math
import tensorflow as tf
from tensorflow.python.saved_model.simple_save import simple_save

def train(X_train, Y_train, X_test, Y_test, path_to_model, learning_rate=0.01, iterations=1500, minibatch_size=500, print_cost=True):

    def placeholders(X_height, X_weight, Y_height, channels):
        X = tf.placeholder(tf.float32, shape=[None, X_height, X_weight, channels], name="myInput")
        Y = tf.placeholder(tf.float32, shape=[None, Y_height], name="myOutput")
        return X, Y

    def inititalize():
        W1 = tf.get_variable("W1", [3, 3, 1, 64], initializer=tf.contrib.layers.xavier_initializer())
        W2 = tf.get_variable("W2", [2, 2, 64, 128], initializer=tf.contrib.layers.xavier_initializer())

        nn_weights = {"W1": W1, "W2": W2}
        return nn_weights

    def forward_propagation(X, nn_weights):
        W1 = nn_weights['W1']
        W2 = nn_weights['W2']

        C1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
        C1 = fc1 = tf.nn.relu(C1)
        M1 = tf.nn.max_pool(C1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        C2 = tf.nn.conv2d(M1, W2, strides=[1, 1, 1, 1], padding='SAME')
        C2 = tf.nn.relu(C2)
        M2 = tf.nn.max_pool(C2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        FLATTEN = tf.contrib.layers.flatten(M2)
        Z3 = tf.contrib.layers.fully_connected(FLATTEN, 2, activation_fn=None)
        return Z3

    def compute_cost(Z3, Y):
       return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))

    def backward_propagation(learning_rate, cost):
        return tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    def random_mini_batches(X, Y, mini_batch_size=5000):

        m = X.shape[0]
        mini_batches = []
        permutation = list(np.random.permutation(m))
        shuffled_X = X[permutation, :, :]
        shuffled_Y = Y[permutation, :]
        num_complete_minibatches = math.floor(
        m / mini_batch_size)
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :]
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :]
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches

    def save_model(sess,path_to_model,X,Y):
        try:
            shutil.rmtree(path_to_model)
        except:
            pass
        simple_save(sess, path_to_model, inputs={"myInput": X}, outputs={"myOutput": Y})

    def create_model(X_train, Y_train, X_test, Y_test, path_to_model, learning_rate=0.001,
              iterations=1500, minibatch_size=500):

        tf.reset_default_graph()
        (num_labels, X_height, X_weight, channels) = X_train.shape
        Y_height = Y_train.shape[1]

        X, Y = placeholders(X_height, X_weight, Y_height,channels)
        parameters = inititalize()
        Z3 = forward_propagation(X, parameters)
        cost = compute_cost(Z3, Y)
        optimizer = backward_propagation(learning_rate, cost)
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            for iteration in range(iterations):
                iteration_cost = 0.
                num_minibatches = int(num_labels / minibatch_size)
                minibatches = random_mini_batches(X_train, Y_train, minibatch_size)
                for minibatch in minibatches:
                    (minibatch_X, minibatch_Y) = minibatch
                    _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                    iteration_cost += minibatch_cost / num_minibatches
                if print_cost == True and iteration % 100 == 0:
                    print("Loss after iteration " + str(iteration) + ": " + str(iteration_cost))
            predict_op = tf.argmax(Z3, 1)
            correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Train:", accuracy.eval({X: X_train, Y: Y_train}))
            print("Test:", accuracy.eval({X: X_test, Y: Y_test}))

            save_model(sess,path_to_model, X, Y)

    create_model(X_train, Y_train, X_test, Y_test, path_to_model, learning_rate, iterations, minibatch_size)