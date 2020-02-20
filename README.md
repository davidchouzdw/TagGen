# Introduction
TagGen an end-to-end deep generative framework named TagGen, that addresses all the above challenges. In particular, we first design a novel sampling strategy to jointly extract structural and temporal context information directly from a collection of timestamped edges. Then, inspired by the advances of Transformer model in symbolic sequence modeling, TagGen parameterizes a bi-level self-attention mechanism together with a family of local operations to generate temporal random walks. At last, a discriminator gradually selects generated temporal random walks, that are plausible in the input data, and feeds them to an assembling module for generating temporal networks.


# Paper
A Data Driven Graph Generative Model for Temporal Interaction Networks

# Requirements
* python 3.7
* pytorch 1.2

# Command for Training
python graph_fairnet.py -d DBLP -w 5 -t 15 -b -g 0 -m

# Command for Test
python graph_fairnet.py -d DBLP -w 5 -t 15 -b -g 0
