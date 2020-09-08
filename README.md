## A Data-Driven Graph Generative Model for Temporal Interaction Networks

### Introduction
TagGen is an end-to-end deep generative model that is able to directly learn from the raw while finest-grained temporal 
networks (i.e., a collection of time-stamped edges). In particular, our framework is built based on a Transformer machine 
that learns the distribution of temporal random walks over the input data. To mimic the dynamic systems, TagGen is 
equipped with a novel context generation scheme that defines a family of local operations to perform addition and 
deletion over nodes and edges dynamically.


### Requirement:
* python 3.7
* pytorch 1.6 with gpu (cuda 10.2)

### Environment and Installation:
1. conda env create -f environment.yml
2. conda activate TagGen

### Command
1. Training:
python graph_fairnet.py -d DBLP -w 5 -t 15 -b -g 0 -m

2. Testing and evaluation:
python graph_fairnet.py -d DBLP -w 5 -t 15 -b -g 0


### Some important parameters:
* -d: the path of input graph
* -g: the index of the gpu, 0 is the default value. If not using gpu, ignore this flag.
* -t: the number of timestamps or time slices
* -w: time windows sizes. A node could only connect to another node if the difference between their timestamp is
 within this range.
* -b: the biased temporal random walk or unbaised temporal random walk. Biased temporal random walk depends on the node 
proximity, while unbiased temporal random walk is independent of node proximity. The default value is biased random walk
 with this flag.
* -m: the training mode or test mode. the default value is training mode with this flag.


### Evaluation：
The final results will be stored in the directory: "./data/DBLP/metrics.txt".
The synthetic graph is stored in the directory:"./data/DBLP/DBLP_output_sequences.txt".


### Reference:
@inproceedings{zhou2020data,
  title={A Data-Driven Graph Generative Model for Temporal Interaction Networks},
  author={Zhou, Dawei and Zheng, Lecheng and Han, Jiawei and He, Jingrui},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={401--411},
  year={2020}
}
