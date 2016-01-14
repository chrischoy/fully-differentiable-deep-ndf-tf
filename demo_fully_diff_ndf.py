import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data

DEPTH = 3
N_LEAF = 2 ** (DEPTH + 1)
N_LABEL = 10
N_TREE = 5

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def init_prob_weights(shape, minval=-5, maxval=5):
    return tf.Variable(tf.random_uniform(shape, minval, maxval))


def model(X, w, w2, w3, w4, w_d, w_l, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w, [1, 1, 1, 1], 'SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, [1, 1, 1, 1], 'SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, [1, 1, 1, 1], 'SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    decision_p = tf.nn.sigmoid(tf.matmul(l4, w_d))
    leaf_p = tf.nn.softmax(w_l)

    return decision_p, leaf_p


mnist = input_data.read_data_sets("MNIST/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28, 1)
teX = teX.reshape(-1, 28, 28, 1)

N_BATCH = 128
X = tf.placeholder("float", [N_BATCH, 28, 28, 1])
Y = tf.placeholder("float", [N_BATCH, N_LABEL])

w = init_weights([3, 3, 1, 32])
w2 = init_weights([3, 3, 32, 64])
w3 = init_weights([3, 3, 64, 128])
w4 = init_weights([128 * 4 * 4, 625])

w_d = init_prob_weights([625, N_LEAF], -1, 1)
w_l = init_prob_weights([N_LEAF, N_LABEL], -2, 2)

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

# With the probability decision_p, route a sample to the right branch
decision_p, leaf_p = model(X, w, w2, w3, w4, w_d, w_l, p_keep_conv, p_keep_hidden)

# Compute 1 - d, 1 - \sigmoid (fully connected output)
decision_p_comp = tf.sub(tf.ones_like(decision_p), decision_p)

# Concatenate both d, 1-d
decision_p_pack = tf.pack([decision_p, decision_p_comp])

# Flatten/vectorize the decision, used for indexing.
flat_decision_p = tf.reshape(decision_p_pack, [-1])

# Since we are using batch, 0 index of each data instance is essential in
# finding indices.
batch_0_indices = tf.tile(tf.expand_dims(tf.range(0, N_BATCH * N_LEAF, N_LEAF), 1), [1, N_LEAF])
in_repeat = N_LEAF / 2
out_repeat = N_BATCH
batch_complement_indices = np.array([[0] * in_repeat, [N_BATCH * N_LEAF] * in_repeat] * out_repeat).reshape(N_BATCH, N_LEAF)

# First root node for each data instance.
mu_ = tf.gather(flat_decision_p, tf.add(batch_0_indices, batch_complement_indices))

# from the second layer to the last layer, we make the decision nodes
for d in xrange(1, DEPTH + 1):
    indices = tf.range(2 ** d, 2 ** (d + 1)) - 1
    tile_indices = tf.reshape(tf.tile(tf.expand_dims(indices, 1), [1, 2 ** (DEPTH - d + 1)]), [1, -1])
    batch_indices = tf.add(batch_0_indices, tf.tile(tile_indices, [N_BATCH, 1]))

    in_repeat = in_repeat / 2
    out_repeat = out_repeat * 2
    batch_complement_indices = np.array([[0] * in_repeat, [N_BATCH * N_LEAF] * in_repeat] * out_repeat).reshape(N_BATCH, N_LEAF)

    mu_ = tf.mul(mu_, tf.gather(flat_decision_p, tf.add(batch_indices, batch_complement_indices)))

# Final \mu
mu = mu_

# p(y|x)
py_x = tf.reduce_mean(tf.mul(tf.tile(tf.expand_dims(mu, 2), [1, 1, N_LABEL]), tf.tile(tf.expand_dims(leaf_p, 0), [N_BATCH, 1, 1])), 1)  # average all the leaf p

# cross entropy loss
cost = tf.reduce_mean(-tf.mul(tf.log(py_x), Y))

# cost = tf.reduce_mean(tf.nn.cross_entropy_with_logits(py_x, Y))
train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict = tf.argmax(py_x, 1)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(100):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        sess.run(train_step, feed_dict={X: trX[start:end], Y: trY[start:end], p_keep_conv: 0.8, p_keep_hidden: 0.5})

    results = []
    for start, end in zip(range(0, len(teX), 128), range(128, len(teX), 128)):
        results.extend(np.argmax(teY[start:end], axis=1) == sess.run(predict, feed_dict={X: teX[start:end], p_keep_conv: 1.0, p_keep_hidden: 1.0}))

    print i, np.mean(results)
