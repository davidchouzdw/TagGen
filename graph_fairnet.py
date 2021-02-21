import torch
from torch import nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
from copy import deepcopy
from sklearn.metrics import accuracy_score
import argparse
from graph_utils import *
import scipy.io as sio
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import time


class Config(object):
    # parameters for transformer
    N = 1
    d_model = 80
    d_ff = 128
    h = 4
    dropout = 0.2
    output_size = 2
    lr = 0.003
    max_epochs = 30
    batch_size = 64
    # max number of nodes
    max_sen_len = 25
    # prob for action [insert, delete, skip]
    action_prob = [0.45, 0.35, 0.2]
    search_size = 500
    sample_time = 20
    # parameter for accelerating the computation
    windows_size = 2



def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    "Implementation of Scaled dot product attention"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, 0.00000001)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class BiLevelMultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(BiLevelMultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.coef = 0.5

    def forward(self, x_1, x_2, mask=None):
        "Implements Multi-head attention"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = x_1.size(0)
        query_r, key_r, value_r = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (x_1, x_1, x_1))]
        query_n, key_n, value_n = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (x_2, x_2, x_2))]

        x_r, attn_r = attention(query_r, key_r, value_r, mask=mask,
                                 dropout=self.dropout)
        x_n, attn_n = attention(query_n, key_n, value_n, mask=mask,
                                 dropout=self.dropout)
        x = x_r * (1 - self.coef) + x_n * self.coef
        self.attn = attn_r * (1 - self.coef) + attn_n * self.coef
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class Encoder(nn.Module):
    '''
    Transformer Encoder

    It is a stack of N layers.
    '''

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    '''
    An encoder layer

    Made up of self-attention and a feed forward layer.
    Each of these sublayers have residual and layer norm, implemented by SublayerOutput.
    '''

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.res_0 = Resnet_0(size, dropout)
        self.res_1 = Resnet_1(size, dropout)
        self.size = size

    def forward(self, x_1, x_2, mask=None):
        "Transformer Encoder"
        x = self.res_0(x_1, x_2, lambda x_1, x_2: self.self_attn(x_1, x_2, mask))  # Encoder self-attention
        return self.res_1(x, self.feed_forward)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        "Implements FFN equation."
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class LayerNorm(nn.Module):
    "Construct a layer normalization module."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Resnet_0(nn.Module):
    '''
    A residual connection followed by a layer norm.
    '''

    def __init__(self, size, dropout):
        super(Resnet_0, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_1, x_2, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x_1 + self.dropout(sublayer(self.norm(x_1), self.norm(x_2)))


class Resnet_1(nn.Module):
    '''
    A residual connection followed by a layer norm.
    '''

    def __init__(self, size, dropout):
        super(Resnet_1, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config
        h, N, dropout = config.h, config.N, config.dropout
        d_model, d_ff = config.d_model, config.d_ff
        attn = BiLevelMultiHeadedAttention(h, d_model)
        ff = FeedForward(d_model, d_ff, dropout)
        self.norm = LayerNorm(d_model)
        self.encoder = Encoder(EncoderLayer(config.d_model, deepcopy(attn), deepcopy(ff), dropout), N)
        # Fully-Connected Layer
        self.fc = nn.Linear(d_model, config.output_size)


    def forward(self, x_1, x_2):
        encoded_sents = self.encoder(x_1, x_2)
        final_feature_map = encoded_sents[:, -1, :]
        final_out = self.fc(final_feature_map)
        return F.softmax(final_out, dim=-1)


class data_loader():
    def __init__(self, output_directory, embedding, embedding2, stage='train', mode=False):
        self.role = []
        self.node = []
        self.label = []
        if mode:
            role_level_emb = dict()
            with open(embedding, 'r') as f_emb:
                next(f_emb)
                for line in f_emb:
                    line = line.split()
                    role_level_emb[line[0]] = np.array(list(map(float, line[1:])))
            node_level_emb = dict()
            with open(embedding2, 'r') as f_emb:
                next(f_emb)
                for line in f_emb:
                    line = line.split()
                    node_level_emb[line[0]] = np.array(list(map(float, line[1:])))
            with open('{}/sequences.txt'.format(output_directory), 'r') as f:
                for line in f:
                    line = line.rstrip("\n")
                    nodes = list(map(int, line.split(',')))
                    role_embed = []
                    node_embed = []
                    for node in nodes:
                        role_embed.append(role_level_emb[str(node)])
                        node_embed.append(node_level_emb[str(node)])
                    self.role.append(role_embed)
                    self.node.append(node_embed)
                    self.label.append([0, 1])
            f.close()
            with open('{}/sequences_negative.txt'.format(output_directory), 'r') as f:
                for line in f:
                    line = line.rstrip("\n")
                    nodes = list(map(int, line.split(',')))
                    role_embed = []
                    node_embed = []
                    for node in nodes:
                        role_embed.append(role_level_emb[str(node)])
                        node_embed.append(node_level_emb[str(node)])
                    self.role.append(role_embed)
                    self.node.append(node_embed)
                    self.label.append([1, 0])
            f.close()
            self.role = np.array(self.role)
            self.node = np.array(self.node)
            self.label = np.array(self.label)
            skf = StratifiedKFold(n_splits=5)
            for train_index, test_index in skf.split(self.role, self.label[:, 0]):
                role_train = self.role[train_index]
                role_val = self.role[test_index]
                node_train = self.node[train_index]
                node_val = self.node[test_index]
                y_train = self.label[train_index]
                y_val = self.label[test_index]
                break
            train_data = {'role': role_train, 'label': y_train, 'node': node_train}
            val_data = {'role': role_val, 'label': y_val, 'node': node_val}
            sio.savemat('{}/train.mat'.format(output_directory), train_data)
            sio.savemat('{}/val.mat'.format(output_directory), val_data)
            self.role = []
            self.node = []
            with open('{}/sequences_generation.txt'.format(output_directory), 'r') as f:
                self.node_list = []
                for line in f:
                    line = line.rstrip("\n")
                    nodes = list(map(int, line.split(',')))
                    role_embed = []
                    node_embed = []
                    sequences = []
                    for node in nodes:
                        role_embed.append(role_level_emb[str(node)])
                        node_embed.append(node_level_emb[str(node)])
                        sequences.append(node)
                    self.role.append(role_embed)
                    self.node.append(node_embed)
                    self.node_list.append(sequences)
                generation_data = {'emb': self.role, 'data': self.node_list, 'node_emb': self.node}
                generation_data['emb'] = np.array(generation_data['emb'])
                sio.savemat('{}/generation.mat'.format(output_directory), generation_data)
            f.close()
        if stage == 'train' or stage == 'Train':
            data = sio.loadmat('{}/train.mat'.format(output_directory))
            self.label = data['label']
            self.role = data['role']
            self.node = data['node']
        elif stage == 'val' or stage == 'Val':
            data = sio.loadmat('{}/val.mat'.format(output_directory))
            self.label = data['label']
            self.role = data['role']
            self.node = data['node']
        else:
            raise (NameError('The stage should be either train or test'))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        x_1 = self.role[idx]
        x_2 = self.node[idx]
        y = self.label[idx]
        sample = {'role': np.array(x_1), 'node': np.array(x_2), 'label': y}
        return sample


def run_epoch(train_iterator, val_iterator, epoch, model):
    train_losses = []
    losses = []
    optimizer = torch.optim.SGD(model.parameters(), lr=model.learning_rate, momentum=0.7)
    criteria = F.binary_cross_entropy_with_logits
    start_time = time.time()
    for k in range(epoch):
        for i, batch in enumerate(train_iterator):
            optimizer.zero_grad()
            x_1 = batch['role'].double().to(model.device)
            x_2 = batch['node'].double().to(model.device)
            y = batch['label'].double().to(model.device)
            y_pred = model(x_1, x_2)
            loss = criteria(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            optimizer.step()
            if (i+1) % 100 == 0:
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                losses = []
                val_accuracy = evaluate_model(model, val_iterator)
                print("Epoch: [{}/{}],  iter: {}, average training loss: {:.5f}, val accuracy: {:.4f}, training time = {:.4f}".format(
                    k+1, epoch, i + 1, avg_train_loss, val_accuracy, time.time() - start_time))
                model.train()


def threshold(train_iterator, model):
    all_preds = []
    for idx, batch in enumerate(train_iterator):
        x_1 = batch['role'].double().to(model.device)
        x_2 = batch['node'].double().to(model.device)
        y_pred = model(x_1, x_2)
        predicted = y_pred.cpu().data[:, 1]
        indices = torch.max(y_pred.cpu().data, 1)[1]
        all_preds.extend(predicted[np.where(indices == 1)])
    return sum(all_preds)/len(all_preds)


def evaluate_model(model, iterator):
    all_preds = []
    all_y = []
    for idx, batch in enumerate(iterator):
        x_1 = batch['role'].double().to(model.device)
        x_2 = batch['node'].double().to(model.device)
        y_pred = model(x_1, x_2)
        predicted = torch.max(y_pred.cpu().data, 1)[1]
        all_preds.extend(predicted.numpy())
        all_y.extend(np.array([0 if i[0] else 1 for i in batch['label'].numpy()]))
    score = accuracy_score(all_y, all_preds)
    return score


def generate_sequence(config, output_directory, model):
    sequences_role_emb = sio.loadmat('{}/generation.mat'.format(output_directory))['emb']
    sequences_node_emb = sio.loadmat('{}/generation.mat'.format(output_directory))['node_emb']
    node_sequences = sio.loadmat('{}/generation.mat'.format(output_directory))['data']
    role_level_emb = dict()
    node_level_emb = dict()
    with open(config.embedding, 'r') as f_emb:
        next(f_emb)
        for line in f_emb:
            line = line.split()
            role_level_emb[line[0]] = np.array(list(map(float, line[1:])))
    with open(config.node_embedding, 'r') as f_emb:
        next(f_emb)
        for line in f_emb:
            line = line.split()
            node_level_emb[line[0]] = np.array(list(map(float, line[1:])))
    f_emb.close()
    start_time = time.time()
    for i in range(len(sequences_role_emb)):
        if i % 100 == 0:
            print('Generating {} sequences in {} seconds'.format(i, time.time() - start_time))
        sequence_role_level = sequences_role_emb[i].reshape(1, sequences_role_emb[i].shape[0], config.d_model)
        sequence_node_level = sequences_node_emb[i].reshape(1, sequences_node_emb[i].shape[0], config.d_model)
        node_sequence = node_sequences[i]
        pos = 1
        key_nodes = [0 for _ in range(len(node_sequence)-1)]
        key_nodes.insert(0, 1)
        action = 0
        min_length = min(len(node_sequence), 5)
        for j in range(config.sample_time):
            ind = random.randint(0, sequence_role_level.shape[1] - 1)
            if ind == 0 or ind == len(key_nodes):
                ind = random.randint(0, sequence_role_level.shape[1] - 1)
            # insertion (action:0)
            if action == 0:
                # sampling nodes
                sampling_nodes = np.random.permutation(range(len(role_level_emb)))[:config.search_size]
                sampling_role_level_emb = [role_level_emb[str(node)] for node in sampling_nodes]
                sampling_node_level_emb = [node_level_emb[str(node)] for node in sampling_nodes]
                n = len(sampling_nodes)
                candidate_key_nodes = np.zeros((n, len(key_nodes) + 1), dtype=np.int32)
                candidate_sequence_role_level = np.zeros((n, len(key_nodes) + 1, config.d_model))
                candidate_sequence_node_level = np.zeros((n, len(key_nodes) + 1, config.d_model))
                for k in range(n):
                    candidate_key_nodes[k] = np.concatenate([key_nodes[:ind], [0], key_nodes[ind:]])
                    candidate_sequence_role_level[k] = np.concatenate(
                        [sequence_role_level[0, :ind], sampling_role_level_emb[k].reshape(1, config.d_model), sequence_role_level[0, ind:]], axis=0)
                    candidate_sequence_node_level[k] = np.concatenate(
                        [sequence_node_level[0, :ind], sampling_node_level_emb[k].reshape(1, config.d_model), sequence_node_level[0, ind:]], axis=0)
                candidate_sequence_role_level = torch.from_numpy(candidate_sequence_role_level).to(model.device)
                candidate_sequence_node_level = torch.from_numpy(candidate_sequence_node_level).to(model.device)
                if candidate_sequence_role_level.shape[1] <= 5:
                    y_pred = model(candidate_sequence_role_level, candidate_sequence_node_level)
                else:
                    if ind + config.windows_size > sequence_role_level.shape[1] - 1:
                        end_ind = sequence_role_level.shape[1]
                        start_ind = end_ind - 1 - config.windows_size * 2
                    else:
                        start_ind = min(0, ind + config.windows_size)
                        end_ind = start_ind + config.windows_size * 2 + 1
                    y_pred = model(candidate_sequence_role_level[:, start_ind:end_ind, :], candidate_sequence_node_level[:, start_ind:end_ind, :])
                accept_prob = y_pred.cpu().data[:, 1]
                max_accept_prob, indices = torch.max(accept_prob, 0)
                if max_accept_prob > config.threshold:
                    sequence_role_level = candidate_sequence_role_level[indices].cpu().numpy().reshape(1, len(key_nodes) + 1, config.d_model)
                    sequence_node_level = candidate_sequence_node_level[indices].cpu().numpy().reshape(1, len(key_nodes) + 1, config.d_model)
                    node_sequence = np.concatenate(
                        [node_sequence[:ind], [sampling_nodes[indices]], node_sequence[ind:]], axis=0)
                    key_nodes = candidate_key_nodes[indices]
                else:
                    action = 2
            # deletion (action: 1)
            if action == 1:
                # avoid deleting key nodes
                if key_nodes[ind] == 1.0:
                    continue
                if len(key_nodes) <= min_length:
                    continue
                else:
                    candidate_sequence_role_level = np.zeros((len(key_nodes), len(key_nodes) - 1, config.d_model))
                    candidate_sequence_node_level = np.zeros((len(key_nodes), len(key_nodes) - 1, config.d_model))
                    candidate_key_nodes = [[] for _ in range(len(key_nodes))]
                    for k in range(len(key_nodes)):
                        candidate_key_nodes[k] = np.concatenate([key_nodes[:k], key_nodes[k + 1:]], axis=0)
                        candidate_sequence_role_level[k] = np.concatenate([sequence_role_level[0, :k], sequence_role_level[0, k + 1:]], axis=0)
                        candidate_sequence_node_level[k] = np.concatenate([sequence_node_level[0, :k], sequence_node_level[0, k + 1:]], axis=0)
                    candidate_sequence_role_level = torch.from_numpy(candidate_sequence_role_level).to(model.device)
                    candidate_sequence_node_level = torch.from_numpy(candidate_sequence_node_level).to(model.device)
                    y_pred = model(candidate_sequence_role_level, candidate_sequence_node_level)
                    accept_prob = y_pred.cpu().data[:, 0]
                    max_accept_prob, indices = torch.max(accept_prob, 0)
                    if max_accept_prob > config.threshold and choose_action([0.5, 0.5]):
                        sequence_role_level = candidate_sequence_role_level[indices].cpu().numpy().reshape(1, len(key_nodes)-1, config.d_model)
                        sequence_node_level = candidate_sequence_node_level[indices].cpu().numpy().reshape(1, len(key_nodes)-1, config.d_model)
                        key_nodes = candidate_key_nodes[indices]
                        node_sequence = np.concatenate([node_sequence[:indices], node_sequence[indices + 1:]], axis=0)
                    else:
                        action = 2
            if action == 2:
                pos += 1
            action = choose_action(config.action_prob)
            if len(key_nodes) >= config.max_sen_len:
                break
            if len(key_nodes) <= min_length:
                break
        with open('{}'.format(config.use_output_path), 'a') as g:
            g.write(', '.join(map(str, node_sequence)) + '\n')


def main(args, config, output_directory):
    if args.mode:
        train_dataset = data_loader(output_directory, config.embedding, config.node_embedding, 'train', args.mode)
        train_iterator = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)
        device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
        val_dataset = data_loader(output_directory, config.embedding, config.node_embedding,  'val')
        val_iterator = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)
        model = Transformer(config).to(device)
        model = model.double()
        model.device = device
        model.learning_rate = 0.08
        run_epoch(train_iterator, val_iterator, config.max_epochs, model)
        print('Finish training process!')
        torch.save(model.state_dict(), '{}/model_epoch_{}.ckpt'.format(output_directory, config.max_epochs))
    else:
        device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
        model = Transformer(config).to(device)
        model = model.double()
        model.device = device
        model.learning_rate = 0.08
        train_dataset = data_loader(output_directory, config.embedding, config.node_embedding,  'train')
        train_iterator = DataLoader(train_dataset, batch_size=50, shuffle=True, num_workers=8)
        config.threshold = max(threshold(train_iterator, model), 0.9)
        generate_sequence(config, output_directory, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("FinTech", conflict_handler='resolve')
    parser.add_argument("-t", dest="slices", type=int, default=15, help="timestamp")
    parser.add_argument('-w', dest='window', type=int, default=5, help='time window sizes')
    parser.add_argument('-d', dest='data', type=str, default='DBLP', help='data directory')
    parser.add_argument('-g', dest='gpu', type=str, default='0', help='the index of GPU')
    parser.add_argument('-b', dest='biased', action='store_true', help="biased or unbiased, default is biased")
    parser.add_argument('-m', dest='mode', action='store_true', help='train or test, default is train')
    args = parser.parse_args()
    config = Config()
    biased = args.biased
    # path of data for training language model
    args.data_path = './data/{}/sequences.txt'.format(args.data)
    # data path of original sentences
    args.embedding = './data/{}/{}_emb'.format(args.data, args.data)
    config.embedding = './data/{}/{}_emb'.format(args.data, args.data)
    config.node_embedding = './data/{}/{}_node_level_emb'.format(args.data, args.data)
    args.model_path = './model_{}/'.format(args.data)
    config.use_output_path = './data/{}/{}_output_sequences.txt'.format(args.data, args.data)
    output_directory = "./data/{}".format(args.data)
    data_directory = './data/{}/edgelist.txt'.format(args.data)
    args.emb_size = config.d_model
    if args.mode:
        interval = args.slices
        data_process(args, interval, biased, time_windows=args.window, data_directory=data_directory, output_directory=output_directory,
                     directed=False)
        main(args, config, output_directory)
    else:
        main(args, config, output_directory)
        os.system('python metrics.py -d {}'.format(args.data))
