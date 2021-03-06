#
#
#  Very simple LSTM implementation.
#
#


import tensorflow as tf
from itertools import chain
from data_utils import load_task, vectorize_sentences

tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for Adam Optimizer.")
tf.flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 13, "Batch size for training.")
tf.flags.DEFINE_integer("epochs", 200, "Number of epochs to train for.")
tf.flags.DEFINE_integer("task_id", 1, "bAbI task id, 1 <= id <= 20")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_integer("num_hidden", 100, "Number of hidden units.")
# tf.flags.DEFINE_integer("steps", 25, "Number of steps in the LSTM.")
tf.flags.DEFINE_string("data_dir", "data/tasks_1-20_v1-2/en/", "Directory containing bAbI tasks")
FLAGS = tf.flags.FLAGS

print("Started Task:", FLAGS.task_id)

# task data
train, test = load_task(FLAGS.data_dir, FLAGS.task_id)
data = train + test

vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

max_story_size = max(map(len, (s for s, _, _ in data)))
sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
query_size = max(map(len, (q for _, q, _ in data)))

num_steps = max_story_size * sentence_size + query_size

S, A = vectorize_sentences(train, word_idx, sentence_size, max_story_size, num_steps)

inputs = 1

input_data = tf.placeholder(tf.float32, [None, num_steps, inputs])
input_label = tf.placeholder(tf.float32, [None, 20])


def inference(_input_data):
    W_o = tf.Variable(tf.random_normal([FLAGS.num_hidden, 20], stddev=0.1), name='W_o')
    b_o = tf.Variable(tf.zeros([1]), name='b_o')

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=FLAGS.num_hidden, forget_bias=0.0,
                                             state_is_tuple=True)  # ,activation="tanh" probar esto

    initial_state = lstm_cell.zero_state(FLAGS.batch_size, tf.float32)
    outputs = []
    state = initial_state
    with tf.variable_scope("RNN"):
        for time_step in range(num_steps):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()
            inp = _input_data[:, time_step, :]
            (cell_output, state) = lstm_cell(inp, state)
            outputs.append(tf.nn.sigmoid(tf.matmul(cell_output, W_o) + b_o))

    result = tf.transpose(tf.squeeze(outputs), perm=[1, 0, 2])
    return result


# Training ------------------------------------------------

predicted = inference(input_data)
cross_entropy = tf.reduce_sum(tf.square(input_label - predicted[:, 62]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
accuracy = tf.reduce_sum(tf.abs(tf.round(predicted[:, 62]) - input_label))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

res = sess.run(cross_entropy, feed_dict={input_data: S[0:13], input_label: A[0:13]})

print res.shape
print S[0:13].shape
print A[0:13].shape

batches = zip(range(0, len(S[0]), FLAGS.batch_size), range(FLAGS.batch_size, len(S[0]), FLAGS.batch_size))
batches = [(start, end) for start, end in batches]


i = 0
for j in xrange(10000):
    for start, end in batches:
        sess.run(train_step, feed_dict={input_data: S[start:end], input_label: A[start:end]})
        i += 1
        if i % 20 == 0:
            print "iteration: ", i, "ce: ", sess.run(cross_entropy,
                                                     feed_dict={input_data: S[start:end], input_label: A[start:end]})


# Test ----------------------------------------------------

# TODO: do the test