# _*_ coding:utf-8 _*_
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from utils import *
import tensorly
import faiss

import scipy.sparse as sp
import time

class TripleFeaturePropagation:

    def __init__(self, initial_feature, train_pair=None):
        self.train_pair = train_pair
        self.initial_feature = initial_feature

    def propagation(self, node_size, rel_size, ent_tuple, triples_idx, ent_ent, ent_ent_val, rel_ent, ent_rel, rel_dim=512, mini_dim=16):

        ent_feature = self.initial_feature

        rel_feature = tf.zeros((rel_size, ent_feature.shape[-1]))

        ent_ent_graph = sp.coo_matrix((ent_ent_val, ent_ent.transpose()), shape=(node_size, node_size))
        ent_ent_graph = normalize_adj(ent_ent_graph)
        ent_ent_graph = convert_sparse_matrix_to_sparse_tensor(ent_ent_graph)

        rel_ent_graph = sp.coo_matrix((K.ones(rel_ent.shape[0]), rel_ent.transpose()), shape=(rel_size, node_size))
        rel_ent_graph = normalize_adj(rel_ent_graph)
        rel_ent_graph = convert_sparse_matrix_to_sparse_tensor(rel_ent_graph)

        ent_rel_graph = sp.coo_matrix((K.ones(ent_rel.shape[0]), ent_rel.transpose()), shape=(node_size, rel_size))
        ent_rel_graph = normalize_adj(ent_rel_graph)
        ent_rel_graph = convert_sparse_matrix_to_sparse_tensor(ent_rel_graph)

        ent_list, rel_list = [ent_feature], [rel_feature]
        start_time = time.time()
        for i in range(1):  # Dual-AMN iteration: 11:81.59, 12:81.62, 13:81.6, .
            new_rel_feature = batch_sparse_matmul(rel_ent_graph, ent_feature)
            new_rel_feature = tf.nn.l2_normalize(new_rel_feature, axis=-1)

            new_ent_feature = batch_sparse_matmul(ent_ent_graph, ent_feature)
            new_ent_feature = new_ent_feature.numpy()

            # ### Keeping stationary for aligned pairs ###
            if self.train_pair.any():
                ori_feature = self.initial_feature.numpy()
                new_ent_feature[self.train_pair[:, 0]] = ori_feature[self.train_pair[:, 0]]
                new_ent_feature[self.train_pair[:, 1]] = ori_feature[self.train_pair[:, 1]]

            new_ent_feature += batch_sparse_matmul(ent_rel_graph, rel_feature)
            new_ent_feature = tf.nn.l2_normalize(new_ent_feature, axis=-1)

            ent_feature = new_ent_feature
            rel_feature = new_rel_feature
            ent_list.append(ent_feature)
            rel_list.append(rel_feature)

        ent_feature = K.l2_normalize(K.concatenate(ent_list, 1), -1)
        rel_feature = K.l2_normalize(K.concatenate(rel_list, 1), -1)
        rel_feature = random_projection(rel_feature, rel_dim)


        batch_size = ent_feature.shape[-1] // mini_dim
        sparse_graph = tf.SparseTensor(indices=triples_idx, values=K.ones(triples_idx.shape[0]),
                                       dense_shape=(np.max(triples_idx) + 1, rel_size))
        adj_value = batch_sparse_matmul(sparse_graph, rel_feature)

        features_list = []
        for batch in range(rel_dim // batch_size + 1):
            temp_list = []
            for head in range(batch_size):
                if batch * batch_size + head >= rel_dim:
                    break
                sparse_graph = tf.SparseTensor(indices=ent_tuple, values=adj_value[:, batch * batch_size + head],
                                               dense_shape=(node_size, node_size))
                feature = batch_sparse_matmul(sparse_graph, random_projection(ent_feature, mini_dim))
                temp_list.append(feature)
            if len(temp_list):
                features_list.append(K.concatenate(temp_list, -1).numpy())
        features = np.concatenate(features_list, axis=-1)
        faiss.normalize_L2(features)

        # ### Test the reconstructed entity feature ####
        features = np.concatenate([ent_feature, features], axis=-1)
        return features
