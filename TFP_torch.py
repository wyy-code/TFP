import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import time
from utils import *  # Assuming this contains necessary utility functions compatible with PyTorch

class TripleFeaturePropagation:

    def __init__(self, train_pair, initial_feature):
        self.train_pair = train_pair
        self.initial_feature = initial_feature

    def propagation(self, node_size, rel_size, ent_tuple, triples_idx, ent_ent, ent_ent_val, rel_ent, ent_rel, rel_dim=512, mini_dim=16):
        start_time = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ent_feature = torch.tensor(self.initial_feature, dtype=torch.float).to(device)

        rel_feature = torch.zeros(rel_size, ent_feature.shape[-1], device=device)

        ent_ent_graph = self._convert_to_sparse_tensor(ent_ent, ent_ent_val, node_size)
        rel_ent_graph = self._convert_to_sparse_tensor(rel_ent, np.ones(rel_ent.shape[0]), rel_size, node_size)
        ent_rel_graph = self._convert_to_sparse_tensor(ent_rel, np.ones(ent_rel.shape[0]), node_size, rel_size)

        ent_list, rel_list = [ent_feature], [rel_feature]

        for i in range(1):  # Iteration loop
            new_rel_feature = torch.sparse.mm(rel_ent_graph, ent_feature)
            new_rel_feature = F.normalize(new_rel_feature, p=2, dim=-1)

            new_ent_feature = torch.sparse.mm(ent_ent_graph, ent_feature).detach().cpu().numpy()

            # Keeping stationary for aligned pairs
            ori_feature = self.initial_feature
            new_ent_feature[self.train_pair[:, 0]] = ori_feature[self.train_pair[:, 0]]
            new_ent_feature[self.train_pair[:, 1]] = ori_feature[self.train_pair[:, 1]]

            new_ent_feature = torch.from_numpy(new_ent_feature).to(device)
            new_ent_feature += torch.sparse.mm(ent_rel_graph, rel_feature)
            new_ent_feature = F.normalize(new_ent_feature, p=2, dim=-1)

            ent_feature = new_ent_feature
            rel_feature = new_rel_feature
            ent_list.append(ent_feature)
            rel_list.append(rel_feature)

        ent_feature = F.normalize(torch.cat(ent_list, 1), p=2, dim=-1)
        rel_feature = F.normalize(torch.cat(rel_list, 1), p=2, dim=-1)
        rel_feature = self._random_projection(rel_feature, rel_dim)

        batch_size = ent_feature.shape[-1] // mini_dim
        sparse_graph = self._create_sparse_tensor(triples_idx, np.ones(triples_idx.shape[0]), rel_size)
        adj_value = torch.sparse.mm(sparse_graph, rel_feature)

        features_list = []
        for batch in range(rel_dim // batch_size + 1):
            temp_list = []
            for head in range(batch_size):
                if batch * batch_size + head >= rel_dim:
                    break
                sparse_graph = self._create_sparse_tensor(ent_tuple, adj_value[:, batch * batch_size + head], node_size)
                feature = torch.sparse.mm(sparse_graph, self._random_projection(ent_feature, mini_dim))
                temp_list.append(feature)
            if len(temp_list):
                features_list.append(torch.cat(temp_list, -1).detach().cpu().numpy())
        features = np.concatenate(features_list, axis=-1)

        faiss.normalize_L2(features)
        print(time.time()-start_time)

        features = np.concatenate([ent_feature.detach().cpu().numpy(), features], axis=-1)
        return features

    def _convert_to_sparse_tensor(self, indices, values, dim1, dim2=None):
        if dim2 is None:
            dim2 = dim1
        indices = torch.tensor(indices, dtype=torch.long).t()
        values = torch.tensor(values, dtype=torch.float)
        return torch.sparse_coo_tensor(indices, values, (dim1, dim2)).to(device)

    def _random_projection(self, tensor, dim):
        # Implement random projection logic
        pass

    def _create_sparse_tensor(self, indices, values, dim):
        # Implement logic to create sparse tensor
        pass

# Usage
# train_pair = ...
# initial_feature = ...
# triple_feature_propagation = TripleFeaturePropagation(train_pair, initial_feature)
# output = triple_feature_propagation.propagation(...)
