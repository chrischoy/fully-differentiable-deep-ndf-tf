# Fully Differentiable Deep Neural Decision Forest

[![DOI](https://zenodo.org/badge/20267/chrischoy/fully-differentiable-deep-ndf-tf.svg)](https://zenodo.org/badge/latestdoi/20267/chrischoy/fully-differentiable-deep-ndf-tf)

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


