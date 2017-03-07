#
#
#  Very simple LSTM implementation.
#
#


import tensorflow as tf
import sampledata

batch_size = 10
digits = num_steps = 8
inputs = 2
num_hidden = 5


data, label = sampledata.create_data(batch_size)

input_data = tf.placeholder(tf.float32, [None, num_steps, inputs])
input_label = tf.placeholder(tf.float32, [None, digits])


def inference(_input_data):
    W_o = tf.Variable(tf.random_normal([num_hidden, 1], stddev=0.1), name='W_o')
    b_o = tf.Variable(tf.zeros([1]), name='b_o')

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden, forget_bias=0.0,
                                             state_is_tuple=True)  # ,activation="tanh" probar esto

    # gru_cell = tf.contrib.rnn.GRUCell(num_units=num_hidden)  # ,activation="tanh" probar esto

    initial_state = lstm_cell.zero_state(batch_size, tf.float32)
    outputs = []
    state = initial_state
    with tf.variable_scope("RNN"):
        for time_step in range(num_steps):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()
            inp = _input_data[:, time_step, :]
            (cell_output, state) = lstm_cell(inp, state)
            outputs.append(tf.nn.sigmoid(tf.matmul(cell_output, W_o) + b_o))

    result = tf.transpose(tf.squeeze(outputs))
    return result


# Training ------------------------------------------------

predicted = inference(input_data)
cross_entropy = tf.reduce_sum(tf.square(input_label - predicted))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
accuracy = tf.reduce_sum(tf.abs(tf.round(predicted) - input_label))

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
print "Result:", sess.run(tf.round(predicted), feed_dict={input_data: data})
