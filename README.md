# Fully Differentiable Deep Neural Decision Forest

[![DOI](https://zenodo.org/badge/20267/chrischoy/fully-differentiable-deep-ndf-tf.svg)](https://zenodo.org/badge/latestdoi/20267/chrischoy/fully-differentiable-deep-ndf-tf)

This repository contains a simple modification of the deep-neural decision
forest [Kontschieder et al.] in TensorFlow. The modification allows joint
optimization of the decision nodes and leaf nodes which theoretically should speed up the training
(haven't verified).


## Motivation:

Deep Neural Deicision Forest, ICCV 2015, proposed an interesting way to incorporate a decision forest into a neural network.

The authors proposed incorporating the terminal nodes of a decision forest as static probability distributions and routing probabilities using sigmoid functions. The final loss is defined as the usual cross entropy between ground truth and weighted average of the terminal probabilities (weights being the routing probabilities).

As there are two trainable parameters, the authors used alternating optimization. They first fixed the terminal node probabilities and trained the base network (routing probabilities), then, fixed the network and optimized the terminal nodes. Such alternating optimization is usually slower than joint optimization since variables that are not being optimized slow down the optimization of the other variable.

However, if we parametrize the terminal nodes using a parametric probability distribution, we can jointly train both terminal and decision nodes, and theoretically, can speed up the convergence.

This code is just a proof-of-concept that

1. One can train both decision nodes and leaf nodes $\pi$ jointly using parametric formulation of leaf (terminal) nodes.

2. It is easy to implement such idea in a symbolic math library.


## Formulation

The leaf node probability $p \in \Delta^{n-1}$ can be parametrized using an $n$ dimensional vector $w_{leaf}$ $\exists w_{leaf}$ s.t. $p = softmax(w_{leaf})$. Thus, we can compute the gradient of $L$ w.r.t $w_{leaf}$ as well and can jointly optimize the terminal nodes as well.

## Experiment

I used a simple (3 convolution + 2 fc) network for this experiment. On the MNIST, it reaches 99.1% after 10 epochs.

## Slides

[SDL Reading Group Slides](https://docs.google.com/presentation/d/1Ze7BAiWbMPyF0ax36D-aK00VfaGMGvvgD_XuANQW1gU/edit?usp=sharing)


## Reference

[Kontschieder et al.] Deep Neural Decision Forests, ICCV 2015
