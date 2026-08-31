import os
import numpy as np
import linecache
import torch


def _nonempty_lines(path):
    if not path or not os.path.isfile(path) or os.path.getsize(path) == 0:
        return []
    lines = linecache.getlines(path)
    return [line.rstrip('\n') for line in lines]


def _parts(line):
    return line.strip().split()


class Dataset(object):

    def __init__(self, config):
        self.graph_file = config['graph_file']
        self.feature_file = config['feature_file']
        self.label_file = config['label_file']
        self.walks_file = config.get('walks_file') or None
        self.device = config['device']

        self.A, self.X, self.W, self.L, self.num_classes = self._load_data()

        self.num_nodes = self.A.shape[0]
        self.num_feas = self.X.shape[1]
        self.num_edges = np.sum(self.A) / 2

        self.A = torch.tensor(self.A, dtype=torch.float32, device=self.device)
        self.X = torch.tensor(self.X, dtype=torch.float32, device=self.device)
        if self.W.size > 0:
            self.W = torch.tensor(self.W, dtype=torch.float32, device=self.device)
        else:
            self.W = torch.empty(0, device=self.device)
        self.L = torch.tensor(self.L, dtype=torch.float32, device=self.device)
        print('nodes {}, edes {}, features {}, classes {}'.format(self.num_nodes, self.num_edges, self.num_feas, self.num_classes))


    def _load_data(self):
        lines = _nonempty_lines(self.label_file)

        #===========load label============
        node_map = {}
        label_map = {}
        Y = []
        cnt = 0
        for line in lines:
            parts = _parts(line)
            if not parts:
                continue
            node_map[parts[0]] = len(node_map)
            y = []
            for label in parts[1:]:
                if not label:
                    continue
                if label not in label_map:
                    label_map[label] = cnt
                    cnt += 1
                y.append(label_map[label])
            Y.append(y if y else [0])
        num_classes = len(label_map)
        num_nodes = len(node_map)

        L = np.array([la[0] for la in Y])

        #=========load feature==========
        lines = _nonempty_lines(self.feature_file)
        feat_parts = next((_parts(line) for line in lines if _parts(line)), None)
        num_features = (len(feat_parts) - 1) if feat_parts else 0
        X = np.zeros((num_nodes, num_features), dtype=np.float32)

        for line in lines:
            parts = _parts(line)
            if len(parts) < 2 or parts[0] not in node_map:
                continue
            node_id = node_map[parts[0]]
            X[node_id] = np.array([float(x) for x in parts[1:num_features + 1]])

        #==========load graph========
        A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        lines = _nonempty_lines(self.graph_file)
        for line in lines:
            parts = _parts(line)
            if len(parts) < 2 or parts[0] not in node_map or parts[1] not in node_map:
                continue
            idx1 = node_map[parts[0]]
            idx2 = node_map[parts[1]]
            A[idx2, idx1] = 1.0
            A[idx1, idx2] = 1.0

        #=========load walks========
        walk_lines = _nonempty_lines(self.walks_file)
        if walk_lines:
            W = np.zeros((num_nodes, num_nodes), dtype=np.float32)
            for line in walk_lines:
                parts = _parts(line)
                if len(parts) < 2 or parts[0] not in node_map or parts[1] not in node_map:
                    continue
                idx1 = node_map[parts[0]]
                idx2 = node_map[parts[1]]
                W[idx2, idx1] = 1.0
                W[idx1, idx2] = 1.0
        else:
            W = np.zeros((0, 0), dtype=np.float32)

        return A, X, W, L, num_classes
