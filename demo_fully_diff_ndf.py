#!/usr/bin/python
"""
# Fully Differentiable Deep Neural Decision Forest

This is an implementation of a simple modification to the deep-neural decision
forest [Kontschieder et al.] usng TensorFlow. The modification allows the joint
optimization of decision nodes and leaf nodes which speeds up the training
(haven't compared yet).


## Motivation:

Deep Neural Deicision Forest, ICCV 2015, proposed a great way to incorporate a
neural network with a decision forest. During the optimization (training), the
terminal (leaf) node has to be updated after each epoch.

This alternating optimization scheme is usually slower than joint optimization
since other variable that is not being optimized slows down the optimization.

This code is just a proof-of-concept that

1. one can train both decision nodes and leaf nodes $\pi$ jointly using
parametric formulation of leaf node.

2. one can implement the idea in a symbolic math library very easily.


## Formulation

The leaf node probability can be parametrized using a $softmax(W_{leaf})$.
i.e. let a vector $W_{leaf} \in \mathbb{R}^N$ where N is the number of classes.

Then taking the soft max operation on W_{leaf} would give us

$$
softmax(W_{leaf}) = \frac{e^{-w_i}}{\sum_j e^{-w_j}}
$$

which is always in a simplex. Thus, without any constraint, we can parametrize
the leaf nodes and can compute the gradient of $L$ w.r.t $W_{leaf}$. This
allows us to jointly optimize both leaf nodes and decision nodes.


## Experiment

Test accuracy after each epoch

```
0 0.955829326923
1 0.979166666667
2 0.982572115385
3 0.988080929487
4 0.988181089744
5 0.988481570513
6 0.987980769231
7 0.989583333333
8 0.991185897436
9 0.991586538462
10 0.991987179487
11 0.992888621795
12 0.993088942308
13 0.992988782051
14 0.992988782051
15 0.992588141026
16 0.993289262821
17 0.99358974359
18 0.992387820513
19 0.993790064103
20 0.994090544872
21 0.993289262821
22 0.993489583333
23 0.99358974359
24 0.993990384615
25 0.993689903846
26 0.99469150641
27 0.994491185897
28 0.994090544872
29 0.994090544872
30 0.99469150641
31 0.994090544872
32 0.994791666667
33 0.993790064103
34 0.994190705128
35 0.994591346154
36 0.993990384615
37 0.995092147436
38 0.994391025641
39 0.993389423077
40 0.994991987179
41 0.994991987179
42 0.994991987179
43 0.994491185897
44 0.995192307692
45 0.995192307692
46 0.994791666667
47 0.995092147436
48 0.994991987179
49 0.994290865385
50 0.994591346154
51 0.994791666667
52 0.995092147436
53 0.995492788462
54 0.994591346154
55 0.995092147436
56 0.994190705128
57 0.99469150641
58 0.99469150641
59 0.994090544872
60 0.994290865385
61 0.994891826923
62 0.994791666667
63 0.994491185897
64 0.994591346154
65 0.994290865385
66 0.99469150641
67 0.994391025641
68 0.994791666667
69 0.99469150641
70 0.994791666667
71 0.994591346154
72 0.994891826923
73 0.994791666667
74 0.995192307692
75 0.995392628205
76 0.995392628205
77 0.995292467949
78 0.994791666667
79 0.995092147436
80 0.995392628205
81 0.994891826923
82 0.995092147436
83 0.994891826923
84 0.995092147436
85 0.995092147436
86 0.995292467949
87 0.994891826923
88 0.995693108974
89 0.994391025641
90 0.994591346154
91 0.995592948718
92 0.995292467949
93 0.995192307692
94 0.994791666667
95 0.995192307692
96 0.995092147436
97 0.994591346154
98 0.995292467949
99 0.995392628205
```

## References
[Kontschieder et al.] Deep Neural Decision Forests, ICCV 2015


## License

The MIT License (MIT)

Copyright (c) 2016 Christopher B. Choy (chrischoy@ai.stanford.edu)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data

DEPTH   = 3                 # Depth of a tree
N_LEAF  = 2 ** (DEPTH + 1)  # Number of leaf node
N_LABEL = 10                # Number of classes
N_TREE  = 5                 # Number of trees (ensemble)
N_BATCH = 128               # Number of data points per mini-batch


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def init_prob_weights(shape, minval=-5, maxval=5):
    return tf.Variable(tf.random_uniform(shape, minval, maxval))


def model(X, w, w2, w3, w4_e, w_d_e, w_l_e, p_keep_conv, p_keep_hidden):
    """
    Create a forest and return the neural decision forest outputs:

        decision_p_e: decision node routing probability for all ensemble
            If we number all nodes in the tree sequentially from top to bottom,
            left to right, decision_p contains
            [d(0), d(1), d(2), ..., d(2^n - 2)] where d(1) is the probability
            of going left at the root node, d(2) is that of the left child of
            the root node.

            decision_p_e is the concatenation of all tree decision_p's

        leaf_p_e: terminal node probability distributions for all ensemble. The
            indexing is the same as that of decision_p_e.
    """
    assert(len(w4_e) == len(w_d_e))
    assert(len(w4_e) == len(w_l_e))

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

    l3 = tf.reshape(l3, [-1, w4_e[0].get_shape().as_list()[0]])
    l3 = tf.nn.dropout(l3, p_keep_conv)

    decision_p_e = []
    leaf_p_e = []
    for w4, w_d, w_l in zip(w4_e, w_d_e, w_l_e):
        l4 = tf.nn.relu(tf.matmul(l3, w4))
        l4 = tf.nn.dropout(l4, p_keep_hidden)

        decision_p = tf.nn.sigmoid(tf.matmul(l4, w_d))
        leaf_p = tf.nn.softmax(w_l)

        decision_p_e.append(decision_p)
        leaf_p_e.append(leaf_p)

    return decision_p_e, leaf_p_e

##################################################
# Load dataset
##################################################
mnist = input_data.read_data_sets("MNIST/", one_hot=True)
trX, trY = mnist.train.images, mnist.train.labels
teX, teY = mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28, 1)
teX = teX.reshape(-1, 28, 28, 1)

# Input X, output Y
X = tf.placeholder("float", [N_BATCH, 28, 28, 1])
Y = tf.placeholder("float", [N_BATCH, N_LABEL])

##################################################
# Initialize network weights
##################################################
w = init_weights([3, 3, 1, 32])
w2 = init_weights([3, 3, 32, 64])
w3 = init_weights([3, 3, 64, 128])

w4_ensemble = []
w_d_ensemble = []
w_l_ensemble = []
for i in range(N_TREE):
    w4_ensemble.append(init_weights([128 * 4 * 4, 625]))
    w_d_ensemble.append(init_prob_weights([625, N_LEAF], -1, 1))
    w_l_ensemble.append(init_prob_weights([N_LEAF, N_LABEL], -2, 2))

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

##################################################
# Define a fully differentiable deep-ndf
##################################################
# With the probability decision_p, route a sample to the right branch
decision_p_e, leaf_p_e = model(X, w, w2, w3, w4_ensemble, w_d_ensemble,
                               w_l_ensemble, p_keep_conv, p_keep_hidden)

flat_decision_p_e = []

# iterate over each tree
for decision_p in decision_p_e:
    # Compute the complement of d, which is 1 - d
    # where d is the sigmoid of fully connected output
    decision_p_comp = tf.sub(tf.ones_like(decision_p), decision_p)

    # Concatenate both d, 1-d
    decision_p_pack = tf.pack([decision_p, decision_p_comp])

    # Flatten/vectorize the decision probabilities for efficient indexing
    flat_decision_p = tf.reshape(decision_p_pack, [-1])
    flat_decision_p_e.append(flat_decision_p)

# 0 index of each data instance in a mini-batch
batch_0_indices = \
    tf.tile(tf.expand_dims(tf.range(0, N_BATCH * N_LEAF, N_LEAF), 1),
            [1, N_LEAF])

###############################################################################
# The routing probability computation
#
# We will create a routing probability matrix \mu. First, we will initialize
# \mu using the root node d, 1-d. To efficiently implement this routing, we
# will create a giant vector (matrix) that contains all d and 1-d from all
# decision nodes. The matrix version of that is decision_p_pack and vectorized
# version is flat_decision_p.
#
# The suffix `_e` indicates an ensemble. i.e. concatenation of all responsens
# from trees.
#
# For depth = 2 tree, the routing probability for each leaf node can be easily
# compute by multiplying the following vectors elementwise.
# \mu =       [d_0,   d_0,   d_0,   d_0, 1-d_0, 1-d_0, 1-d_0, 1-d_0]
# \mu = \mu * [d_1,   d_1, 1-d_1, 1-d_1,   d_2,   d_2, 1-d_2, 1-d_2]
# \mu = \mu * [d_3, 1-d_3,   d_4, 1-d_4,   d_5, 1-d_5,   d_6, 1-d_6]
#
# Tree indexing
#      0
#    1   2
#   3 4 5 6
##############################################################################
in_repeat = N_LEAF / 2
out_repeat = N_BATCH

# Let N_BATCH * N_LEAF be N_D. flat_decision_p[N_D] will return 1-d of the
# first root node in the first tree.
batch_complement_indices = \
    np.array([[0] * in_repeat, [N_BATCH * N_LEAF] * in_repeat]
             * out_repeat).reshape(N_BATCH, N_LEAF)

# First define the routing probabilities d for root nodes
mu_e = []

# iterate over each tree
for i, flat_decision_p in enumerate(flat_decision_p_e):
    mu = tf.gather(flat_decision_p,
                   tf.add(batch_0_indices, batch_complement_indices))
    mu_e.append(mu)

# from the second layer to the last layer, we make the decision nodes
for d in xrange(1, DEPTH + 1):
    indices = tf.range(2 ** d, 2 ** (d + 1)) - 1
    tile_indices = tf.reshape(tf.tile(tf.expand_dims(indices, 1),
                                      [1, 2 ** (DEPTH - d + 1)]), [1, -1])
    batch_indices = tf.add(batch_0_indices, tf.tile(tile_indices, [N_BATCH, 1]))

    in_repeat = in_repeat / 2
    out_repeat = out_repeat * 2

    # Again define the indices that picks d and 1-d for the node
    batch_complement_indices = \
        np.array([[0] * in_repeat, [N_BATCH * N_LEAF] * in_repeat]
                 * out_repeat).reshape(N_BATCH, N_LEAF)

    mu_e_update = []
    for mu, flat_decision_p in zip(mu_e, flat_decision_p_e):
        mu = tf.mul(mu, tf.gather(flat_decision_p,
                                  tf.add(batch_indices, batch_complement_indices)))
        mu_e_update.append(mu)

    mu_e = mu_e_update

##################################################
# Define p(y|x)
##################################################
py_x_e = []
for mu, leaf_p in zip(mu_e, leaf_p_e):
    # average all the leaf p
    py_x_tree = tf.reduce_mean(
        tf.mul(tf.tile(tf.expand_dims(mu, 2), [1, 1, N_LABEL]),
               tf.tile(tf.expand_dims(leaf_p, 0), [N_BATCH, 1, 1])), 1)
    py_x_e.append(py_x_tree)

py_x_e = tf.pack(py_x_e)
py_x = tf.reduce_mean(py_x_e, 0)

##################################################
# Define cost and optimization method
##################################################

# cross entropy loss
cost = tf.reduce_mean(-tf.mul(tf.log(py_x), Y))

# cost = tf.reduce_mean(tf.nn.cross_entropy_with_logits(py_x, Y))
train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict = tf.argmax(py_x, 1)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(100):
    # One epoch
    for start, end in zip(range(0, len(trX), N_BATCH), range(N_BATCH, len(trX), N_BATCH)):
        sess.run(train_step, feed_dict={X: trX[start:end], Y: trY[start:end],
                                        p_keep_conv: 0.8, p_keep_hidden: 0.5})

    # Result on the test set
    results = []
    for start, end in zip(range(0, len(teX), 128), range(128, len(teX), 128)):
        results.extend(np.argmax(teY[start:end], axis=1) ==
            sess.run(predict, feed_dict={X: teX[start:end], p_keep_conv: 1.0,
                                         p_keep_hidden: 1.0}))

    print i, np.mean(results)
