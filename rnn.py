#
#
#  This implements a RNN unrolling the recursive shape of the net and using feedforward nets.
#
#

import tensorflow as tf
import sampledata


batch_size = 10
number_of_neurons = 5
digits = 8

data, label = sampledata.create_data(batch_size)


# Model ------------------------------------------------

W_h = tf.Variable(tf.random_normal([2 + number_of_neurons, number_of_neurons], stddev=0.1), name='W_h')
b_h = tf.Variable(tf.zeros([number_of_neurons]), name='b_h')

W_o = tf.Variable(tf.random_normal([number_of_neurons, 1], stddev=0.1), name='W_o')
b_o = tf.Variable(tf.zeros([1]), name='b_o')

input_data = tf.placeholder(tf.float32, [None, digits, 2])
input_label = tf.placeholder(tf.float32, [None, digits])


def inference(_input_data):

    x = tf.concat(values=[tf.zeros([batch_size, number_of_neurons]), _input_data[:, 0, :]], axis=1)

    output = []

    next_state = tf.nn.sigmoid(tf.matmul(x, W_h) + b_h)
    output.append(tf.nn.sigmoid(tf.matmul(next_state, W_o) + b_o))

    for i in range(1, digits):
        x = tf.concat([next_state, _input_data[:, i, :]], 1)
        next_state = tf.nn.sigmoid(tf.matmul(x, W_h) + b_h)
        output.append(tf.nn.sigmoid(tf.matmul(next_state, W_o) + b_o))

    output = tf.transpose(tf.squeeze(output))

    return output


def predict(_input_data):
    return tf.round(inference(_input_data))


# Training ------------------------------------------------

#cross_entropy = tf.reduce_mean(-tf.reduce_sum(input_label * tf.log(inference(input_data))))
cross_entropy = tf.reduce_sum(tf.square(input_label - inference(input_data)))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

_inference = inference(input_data)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    data, label = sampledata.create_data(batch_size)
    sess.run(train_step, feed_dict={input_data: data, input_label: label})
    if i % 100 == 0:
        print "iteration: ", i, "ce: ", sess.run(cross_entropy, feed_dict={input_data: data, input_label: label})


# Test ----------------------------------------------------

data, label = sampledata.create_data(10)

predicted = predict(input_data)
accuracy = tf.reduce_sum(tf.abs(predicted - label))

print "------------------------------"
print "Number of errors:", sess.run(accuracy, feed_dict={input_data: data, input_label: label})



data = [[[1.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
        [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        [[1.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
        [[1.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
        [[1.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
        [[1.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
        [[1.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
        [[1.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
        [[1.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]]
        ]

print "------------------------------"
print "Result:", sess.run(predicted, feed_dict={input_data: data})
