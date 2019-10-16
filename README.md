# Introduction
FinTech is an end-to-end deep generative model that is able to directly learn from the raw while finest-grained temporal networks (i.e., a collection of time-stamped edges). In particular, our framework is built based on a Transformer machine that learns the distribution of temporal random walks over the input data. To mimic the dynamic systems, FinTech is equipped with a novel context generation scheme that defines a family of local operations to perform addition and deletion over nodes and edges dynamically.


# Paper
Towards Generation of Temporal Networks withFine-Grained Properties


# Requirements
* python 3.7
* pytorch 1.2

# Command for Training
python graph_fairnet.py -d DBLP -w 5 -t 15 -b -g 0 -m

# Command for Test
python graph_fairnet.py -d DBLP -w 5 -t 15 -b -g 0
