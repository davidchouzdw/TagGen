import pickle
import numpy as np
import networkx as nx
from scipy.sparse.csgraph import connected_components
import powerlaw
import time
import argparse


def dictionary_search(dictionary, search_value):
    for key, value in dictionary.items():
        if value == search_value:
            return key


def temporal_metrics_only_FinTech(output_dir, f_out, data_directory_2):
    graph_attr = pickle.load(open(output_dir + '/graph.pickle', "rb"))
    original_network = graph_attr['graph']
    original_node_index = graph_attr['original_index']
    node_index = graph_attr['index']
    n = original_network.shape[0]
    min_time_stamp = np.inf
    max_time_stamp = 0
    with open(output_dir + '/edgelist_new.txt', 'r') as f:
        for line in f:
            line = list(map(int, line.split()))
            if line[2] < min_time_stamp:
                min_time_stamp = line[2]
            if line[2] > max_time_stamp:
                max_time_stamp = line[2]
    windows = max_time_stamp - min_time_stamp + 1
    original_network = np.zeros((windows, n, n), dtype=np.int8)
    with open(output_dir + '/edgelist_new.txt', 'r') as f:
        for line in f:
            line = list(map(int, line.split()))
            a_1 = int(dictionary_search(node_index, line[0]).split('_')[0])
            a_2 = int(dictionary_search(node_index, line[1]).split('_')[0])
            index_i = original_node_index[a_1]
            index_j = original_node_index[a_2]
            for k in range(line[2], max_time_stamp + 1):
                original_network[k, index_i, index_j] = 1
                original_network[k, index_j, index_i] = 1
    for i in range(n):
        for k in range(windows):
            original_network[k, i, i] = 1
    graph = np.zeros((windows, n, n), dtype=np.float16)
    edge_count = [int(np.sum(original_network[k])) for k in range(windows)]
    with open(data_directory_2, 'r+') as f:
        for line in f:
            line = line.rstrip("\n")
            nodes = list(map(int, line.split(',')))
            for i in range(len(nodes) - 1):
                if i <= len(nodes) - 1:
                    a_1 = list(map(int, dictionary_search(node_index, nodes[i]).split('_')))
                    a_2 = list(map(int, dictionary_search(node_index, nodes[i+1]).split('_')))
                    time_stamp = max(a_1[1], a_2[1])
                    index_i = original_node_index[a_1[0]]
                    index_j = original_node_index[a_2[0]]
                    r = np.random.uniform(low=0.85, high=1)
                    for k in range(time_stamp, windows):
                        graph[k, index_i, index_j] += r
                        graph[k, index_j, index_i] += r
    for i in range(n):
        for k in range(windows):
            graph[k, i, i] = graph[k, i, i] + np.random.uniform(low=0.85, high=1)
    for k in range(windows):
        DD = np.sort(graph[k].flatten())[::-1]
        threshold = DD[edge_count[k]]
        graph[k] = np.array(
            [[0 if graph[k, i, j] <= threshold else 1 for i in range(graph.shape[1])]
             for j in range(graph.shape[2])], dtype=np.int8)
    org_graph_metric = []
    our_graph_metric = []
    with open(f_out, 'w') as f:
        for i in range(windows):
            f.write("\n\n\nWhen timestamp = {}\n".format(i))
            aaa = compute_graph_statistics(np.array(original_network[i]), Z_obs=None)
            org_graph_metric.append(aaa)
            f.write('original_graph:\n')
            write_dict(f, aaa)
            aaa = compute_graph_statistics(np.array(graph[i]), Z_obs=None)
            our_graph_metric.append(aaa)
            f.write('\nOurs after training:\n')
            write_dict(f, aaa)
        f.write("\n\n\nOverall Performance:\n")
        header = aaa.keys()
        for metric in header:
            f.write('\nMetric: {}\n'.format(metric))
            org = [item[metric] for item in org_graph_metric]
            mean_median(org, [item[metric] for item in our_graph_metric], f, 'Our')


def average_metric(method_metric, repeated, header, i):
    for metric in header:
        method_metric[i][metric] = method_metric[i][metric] / repeated

def sum_metric(aaa, method_metric, i):
    header = aaa.keys()
    if len(method_metric) <= i:
        method_metric.append(aaa)
    else:
        for metric in header:
            method_metric[i][metric] = method_metric[i][metric] + aaa[metric]


def mean_median(org_graph, generated_graph, f, name):
    org_graph = np.array(org_graph)
    generated_graph = np.array(generated_graph)
    metric = np.divide(np.abs(org_graph - generated_graph), np.abs(org_graph))
    mean = np.mean(metric)
    median = np.median(metric)
    f.write('{}:\n'.format(name))
    f.write('Mean = {}\n'.format(mean))
    f.write('Median = {}\n'.format(median))
    return mean, median


def sampling(network, temporal_graph, n, p=0.5):
    for i in range(n):
        for j in range(n):
            if network[i, j] == 1 and np.random.uniform(low=0.0, high=1) <= p:
                temporal_graph[i, j] = 1


def statistics_degrees(A_in):
    """
    Compute min, max, mean degree

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    d_max. d_min, d_mean
    """

    degrees = A_in.sum(axis=0)
    return np.max(degrees), np.min(degrees), np.mean(degrees)


def statistics_LCC(A_in):
    """
    Compute the size of the largest connected component (LCC)

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Size of LCC

    """

    unique, counts = np.unique(connected_components(A_in)[1], return_counts=True)
    LCC = np.where(connected_components(A_in)[1] == np.argmax(counts))[0]
    return LCC


def statistics_wedge_count(A_in):
    """
    Compute the wedge count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    The wedge count.
    """

    degrees = A_in.sum(axis=0)
    return float(np.sum(np.array([0.5 * x * (x - 1) for x in degrees])))


def statistics_claw_count(A_in):
    """
    Compute the claw count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Claw count
    """

    degrees = A_in.sum(axis=0)
    return float(np.sum(np.array([1 / 6. * x * (x - 1) * (x - 2) for x in degrees])))


def statistics_power_law_alpha(A_in):
    """
    Compute the power law coefficient of the degree distribution of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Power law coefficient
    """

    degrees = A_in.sum(axis=0)
    return powerlaw.Fit(degrees, xmin=max(np.min(degrees), 1)).power_law.alpha


def statistics_gini(A_in):
    """
    Compute the Gini coefficient of the degree distribution of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Gini coefficient
    """

    n = A_in.shape[0]
    degrees = A_in.sum(axis=0)
    degrees_sorted = np.sort(degrees)
    G = (2 * np.sum(np.array([i * degrees_sorted[i] for i in range(len(degrees))]))) / (n * np.sum(degrees)) - (
            n + 1) / n
    return float(G)


def statistics_cluster_props(A, Z_obs):
    def get_blocks(A_in, Z_obs, normalize=True):
        block = Z_obs.T.dot(A_in.dot(Z_obs))
        counts = np.sum(Z_obs, axis=0)
        blocks_outer = counts[:, None].dot(counts[None, :])
        if normalize:
            blocks_outer = np.multiply(block, 1 / blocks_outer)
        return blocks_outer

    in_blocks = get_blocks(A, Z_obs)
    diag_mean = np.multiply(in_blocks, np.eye(in_blocks.shape[0])).mean()
    offdiag_mean = np.multiply(in_blocks, 1 - np.eye(in_blocks.shape[0])).mean()
    return diag_mean, offdiag_mean


def compute_graph_statistics(A_in, Z_obs=None):
    A = A_in.copy()
    A_graph = nx.from_numpy_matrix(A).to_undirected()
    statistics = {}
    start_time = time.time()
    d_max, d_min, d_mean = statistics_degrees(A)
    print("--- %s seconds to compute statistics_degrees ---" % (time.time() - start_time))
    # Degree statistics
    statistics['d'] = d_mean
    # largest connected component
    LCC = statistics_LCC(A)
    print("--- %s seconds to compute statistics_LCC ---" % (time.time() - start_time))
    statistics['LCC'] = LCC.shape[0]
    # wedge count
    statistics['wedge_count'] = statistics_wedge_count(A)
    print("--- %s seconds to compute statistics_wedge_count ---" % (time.time() - start_time))
    # # claw count
    statistics['claw_count'] = statistics_claw_count(A)
    print("--- %s seconds to compute statistics_claw_count ---" % (time.time() - start_time))
    # power law exponent
    statistics['power_law_exp'] = statistics_power_law_alpha(A)
    print("--- %s seconds to compute statistics_power_law_alpha ---" % (time.time() - start_time))
    # Number of connected components
    statistics['n_components'] = connected_components(A, directed=False)[0]
    print("--- %s seconds to compute connected_components ---" % (time.time() - start_time))
    if Z_obs is not None:
        # inter- and intra-community density
        intra, inter = statistics_cluster_props(A, Z_obs)
        statistics['intra_community_density'] = intra
        statistics['inter_community_density'] = inter
    print("--- %s seconds to compute statistics_cluster_props ---" % (time.time() - start_time))
    return statistics


def write_dict(f, aaa):
    for item, key in aaa.items():
        f.write('{} = {}\n'.format(item, key))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Metrics", conflict_handler='resolve')
    parser.add_argument('-d', dest='data', type=str, default='DBLP', help='data directory')
    args = parser.parse_args()
    string = args.data
    output_dir = './data/{}'.format(string)
    filename = '/{}_output_sequences.txt'.format(string)
    file = output_dir + filename[:-4] + '_metric' + filename[-4:]
    data_directory_1 = './data/{}/sequences.txt'.format(string)
    data_directory_2 = './data/{}/{}_output_sequences.txt'.format(string, string)
    temporal_metrics_only_FinTech(output_dir, file, data_directory_2)
