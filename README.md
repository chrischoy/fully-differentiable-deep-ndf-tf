# Fully differentiable deep neural decision forest

This is an implementation of a simple modification to the deep-neural decision
forest usng TensorFlow. The modification allows the joint optimization of
decision and leaf nodes which speeds up the training (haven't compared yet).


## Motivation:

Deep Neural Deicision Forest, ICCV 2015, proposed a great way to incorporate a
neural network with a decision forest. During the optimization (training), the
terminal (leaf) node has to be updated after each epoch.

This alternative optimization scheme is usually slower than joint optimization
since other variable that is not being optimized slows down the optimization.

This code is just a proof-of-concept that

1. one can train both decision nodes and leaf nodes $\pi$ jointly using
parametric formulation of leaf node.

2. one can implement the idea in a symbolic math library very easily.


## Formulation

The leaf node probability can be parametrized using a softmax(W_{leaf}).
i.e. let a vector that has a length # of class be W_{leaf}.

Then taking the soft max operation on W_{leaf} would be

softmax(W_{leaf}) = \frac{e^{-w_i}}{\sum_j e^{-w_j}}

which is always in a simplex. Thus, the gradient of L w.r.t W_{leaf} is
possible and one can jointly optimize both leaf nodes and decision nodes.


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


