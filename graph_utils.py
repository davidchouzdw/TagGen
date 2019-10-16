import numpy as np
import pickle
import os
import random


def graph_similarity(graph):
    alpha = 0.99
    row_sums = graph.sum(axis=1)
    normalized_graph = graph / row_sums[:, np.newaxis]
    x = np.identity(graph.shape[0], dtype=np.int8) - alpha * normalized_graph
    node_similarity = np.linalg.inv(x)
    for i in range(len(node_similarity)):
        node_similarity[i, i] = node_similarity[i, i] / 2
    return node_similarity


def data_process(args, interval, biased, time_windows=2, data_directory='./data/DBLP/edgelist.txt', output_directory="./data/DBLP", directed=False):
    node_dict, node_index, original_node_index, node_value, node_level = preprocess_edgelist(data_directory, interval, directed, time_windows)
    if not os.path.exists("{}/{}_emb".format(output_directory, args.data)):
        os.system("python ./deepwalk/main.py --representation-size {} --input {} --output {}/{}_emb".format(
            args.emb_size, data_directory[:-4] + '_new' + data_directory[-4:], output_directory, args.data))
    # original graph
    graph = np.zeros((len(original_node_index), len(original_node_index)), dtype=np.int8)
    with open(data_directory, 'r+') as f:
        for line in f:
            nodes = list(map(int, line.split()))
            i = original_node_index[nodes[0]]
            j = original_node_index[nodes[1]]
            graph[i, j] = 1
            graph[j, i] = 1
    for i in range(len(original_node_index)):
        graph[i, i] = 1
    # newly-built graph
    new_graph = np.zeros((len(node_index), len(node_index)), dtype=np.float32)
    with open(data_directory[:-4] + '_new' + data_directory[-4:], 'r+') as f:
        for line in f:
            nodes = list(map(int, line.split()))
            new_graph[nodes[0], nodes[1]] = 1.0
            new_graph[nodes[1], nodes[0]] = 1.0
    for i in range(len(node_index)):
        new_graph[i, i] = 1
    degree = np.array(np.sum(new_graph, axis=1), dtype=np.int)
    node_similarity = graph_similarity(new_graph)
    data = dict()
    data['original_index'] = original_node_index
    data['index'] = node_index
    data['value'] = node_value
    data['dict'] = node_dict
    data['graph'] = graph
    data['node_level'] = node_level
    pickle.dump(data, open(output_directory + '/graph.pickle', "wb"))
    role_level_emb = dict()
    with open("{}/{}_emb".format(output_directory, args.data), 'r') as f_emb:
        next(f_emb)
        for line in f_emb:
            line = line.split()
            role_level_emb[line[0]] = np.array(list(map(float, line[1:])))
    with open("{}/{}_node_level_emb".format(output_directory, args.data), 'w') as f_emb:
        node_level_emb = dict()
        f_emb.write('{} {}\n'.format(len(node_level), len(role_level_emb['0'])))
        for i in range(len(node_level)):
            sum = role_level_emb[str(node_level[i][0])]
            for j in range(1, len(node_level[i])):
                sum = sum + role_level_emb[str(node_level[i][j])]
            # node_level_emb[str(i)] = sum/len(node_level[i])
            f_emb.write('{} '.format(i) + ' '.join(map(str, sum/len(node_level[i]))) + '\n')
    temporal_random_walks(node_dict, node_similarity, degree, output_directory, biased, node_index, node_value)


def dictionary_search(dictionary, search_value):
    for key, value in dictionary.items():
        if value == search_value:
            return key


def temporal_random_walks(node_dict, node_similarity, degree, output_directory, biased, node_index, node_value):
    walk_length = 20
    if biased:
        temporal_random_walk = biased_temporal_neighbor_selection_with_prob
        repeated = list(map(int, np.array(degree / 3) + 1))
    else:
        # less sequences but faster speed
        temporal_random_walk = unbiased_temporal_neighbor_selection_with_prob
        repeated = [3 for _ in range(len(degree))]
    count = 0
    with open('{}/sequences.txt'.format(output_directory), 'w+') as f:
        with open('{}/sequences_generation.txt'.format(output_directory), 'w+') as f_g:
            with open('{}/sequences_negative.txt'.format(output_directory), 'w+') as f_neg:
                for item, v in node_dict.items():
                    for j in range(repeated[count]):
                        # adding the initial node to the sequence and initializing the neighbor list
                        sequence = [int(node_index[item])]
                        neighbor_list = v
                        node = item
                        for i in range(1, walk_length):
                            # select neighbors
                            prob_list = temporal_random_walk(neighbor_list, node_similarity, node, node_index)
                            index = choose_action(prob_list)
                            while index is None:
                                index = choose_action(prob_list)
                            node = '{}_{}'.format(neighbor_list[index][0], neighbor_list[index][1])
                            # adding the selected node to the list
                            sequence.append(node_index[node])
                            # reinitialized the neighbor_list
                            neighbor_list = node_dict[node]
                        f.write(', '.join(map(str, sequence)) + '\n')
                        # generate sequence for sequence generation
                        f_g.write(', '.join(map(str, sequence)) + '\n')
                        # generate sequence for negative sample by substituting only 1-3 nodes in the sequence
                        neg_seq = np.array(sequence, copy=True)
                        indices = random.choices(range(len(sequence)), k=random.randint(1, 3))
                        for index in indices:
                            neg_seq[index] = random.randint(0, len(node_dict)-1)
                        f_neg.write(', '.join(map(str, neg_seq)) + '\n')
                    count += 1


# Pr(w) = 1/|Γt (v)| , where the mapping function Γt (v) = T - time_windows (T >= time_windows)
def unbiased_temporal_neighbor_selection_with_prob(neighbor_list, node_similarity=None, k=None, node_index=None):
    """
    input:
    neighbor_list: a list of neighbor stored in the form of a tuple (node, timestamp).
    current_timestamp: the current timestamp

    output:
    prob_list: the probability of each neighbor being chosen in the next temporal random walk
    """
    prob_list = [1.0 / len(neighbor_list) for _ in range(len(neighbor_list))]
    return prob_list


def biased_temporal_neighbor_selection_with_prob(neighbor_list, node_similarity, k, node_index):
    """
    input:
    neighbor_list: a list of neighbor stored in the form of a tuple (node, timestamp).
    current_timestamp: the current timestamp

    output:
    prob_list: the probability of each neighbor being chosen in the next temporal random walk
    """
    prob_list = [node_similarity[node_index[k], node_index['{}_{}'.format(item[0], item[1])]] for item in neighbor_list]
    prob_list = prob_list / np.sum(prob_list)
    return prob_list


def choose_action(c):
    r = np.random.random()
    c = np.array(c)
    for i in range(1, len(c)):
        c[i] = c[i]+c[i-1]
    for i in range(len(c)):
        if c[i] >= r:
            return i


def preprocess_edgelist(data_directory, interval, directed, time_windows):
    node_dict = dict()
    node_index = dict()
    node_value = dict()
    original_node_index = dict()
    count = 0
    original_node_count = 0
    min_time_stamp = np.inf
    max_time_stamp = 0
    # find the minimal timestamp
    with open(data_directory, 'r') as f:
        for line in f:
            line = list(map(int, line.split()))
            if line[2] < min_time_stamp:
                min_time_stamp = line[2]
            if line[2] > max_time_stamp:
                max_time_stamp = line[2]
    time_slice = int((max_time_stamp - min_time_stamp + 1) / interval)
    assert (time_slice != 0), "Please check if interval is correctly set!"
    with open(data_directory, 'r') as f:
        with open(data_directory[:-4] + '_new' + data_directory[-4:], 'w') as f_out:
            for line in f:
                nodes = list(map(int, line.split()))
                nodes[2] = int((nodes[2] - min_time_stamp) / time_slice)
                # index dictionary by which we could map newly-created nodes back to original graph
                if nodes[0] not in original_node_index:
                    original_node_index[nodes[0]] = original_node_count
                    original_node_count += 1
                if nodes[1] not in original_node_index:
                    original_node_index[nodes[1]] = original_node_count
                    original_node_count += 1
                # map original nodes to newly-created nodes to encode the time info
                if '{}_{}'.format(nodes[0], nodes[2]) not in node_index:
                    node_value[count] = '{}_{}'.format(nodes[0], nodes[2])
                    node_index['{}_{}'.format(nodes[0], nodes[2])] = count
                    node_dict['{}_{}'.format(nodes[0], nodes[2])] = [(nodes[0], nodes[2])]
                    count += 1
                # if the second node does not exist in the dictionary, then add it to the dictionary
                if '{}_{}'.format(nodes[1], nodes[2]) not in node_index:
                    node_value[count] = '{}_{}'.format(nodes[1], nodes[2])
                    node_index['{}_{}'.format(nodes[1], nodes[2])] = count
                    node_dict['{}_{}'.format(nodes[1], nodes[2])] = [(nodes[1], nodes[2])]
                    count += 1
                if (nodes[1], nodes[2]) not in node_dict['{}_{}'.format(nodes[0], nodes[2])]:
                    node_dict['{}_{}'.format(nodes[0], nodes[2])].append((nodes[1], nodes[2]))
                if not directed:
                    if (nodes[0], nodes[2]) not in node_dict['{}_{}'.format(nodes[1], nodes[2])]:
                        node_dict['{}_{}'.format(nodes[1], nodes[2])].append((nodes[0], nodes[2]))
                if max_time_stamp < nodes[2]:
                    max_time_stamp = nodes[2]
                f_out.write("{} {} {}\n".format(node_index['{}_{}'.format(nodes[1], nodes[2])], node_index[
                    '{}_{}'.format(nodes[0], nodes[2])], nodes[2]))
                f_out.write("{} {} {}\n".format(node_index['{}_{}'.format(nodes[0], nodes[2])], node_index[
                    '{}_{}'.format(nodes[1], nodes[2])], nodes[2]))

    with open(data_directory[:-4] + '_new' + data_directory[-4:], 'a+') as f_out:
        for node in node_dict.keys():
            nodes = list(map(int, node.split('_')))
            # expand the neighbour of this node if the time stamp is within a certain range
            for i in range(max(0, nodes[1] - time_windows), min(interval, nodes[1] + time_windows)):
                if i != nodes[1] and '{}_{}'.format(nodes[0], i) in node_dict:
                    # if this found neighbor is the node itself, then skip it
                    node_dict[node].append((nodes[0], i))
                    f_out.write("{} {} {}\n".format(node_index[node], node_index['{}_{}'.format(nodes[0], i)], i))
                    f_out.write("{} {} {}\n".format(node_index['{}_{}'.format(nodes[0], i)], node_index[node], i))
    node_level = dict()
    for i in range(len(node_index)):
        node_level[i] = [i]
    for node in node_dict.keys():
        nodes = list(map(int, node.split('_')))
        # expand the neighbour of this node if the time stamp is within a certain range
        for i in range(interval):
            if i != nodes[1] and '{}_{}'.format(nodes[0], i) in node_dict:
                node_level[node_index[node]].append(node_index['{}_{}'.format(nodes[0], i)])
    print("Finish Remapping nodes! Total number of nodes = {}".format(count))
    return node_dict, node_index, original_node_index, node_value, node_level
